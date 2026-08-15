"""FastAPI serving layer for SafeDerm.

Serves whichever model variant is configured (baseline or champion) and,
once 08_calibration_conformal.ipynb has produced a calibration artifact,
serves MC-Dropout-averaged, temperature-scaled confidence with
three-tier routing (src.calibration) instead of raw softmax. Falls back
to raw softmax with `calibrated: false` if no artifact exists yet, or if
the loaded model has no dropout layers to run MC Dropout over -- so this
file doesn't have to wait for 08 to be finished to be deployable.
"""

import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Dict, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.calibration import CalibrationArtifact, assign_tier, mc_dropout_predict
from src.config import BASELINE_MODEL_PATH, CALIBRATION_ARTIFACT_PATH, CHAMPION_MODEL_PATH
from src.labels import ALL_CLASSES, risk_group
from src.model import build_baseline_model, build_champion_model
from src.transforms import get_eval_transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "baseline": {"build_fn": build_baseline_model, "checkpoint_path": BASELINE_MODEL_PATH},
    "champion": {"build_fn": build_champion_model, "checkpoint_path": CHAMPION_MODEL_PATH},
}

MODEL_VARIANT = os.environ.get("SAFEDERM_MODEL_VARIANT", "baseline")
if MODEL_VARIANT not in MODEL_REGISTRY:
    raise ValueError(
        f"Unknown SAFEDERM_MODEL_VARIANT={MODEL_VARIANT!r}; must be one of {list(MODEL_REGISTRY)}"
    )

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# No cookies/sessions/auth anywhere in this API -- allow_credentials has
# nothing to guard, so it stays False (the original True + allow_origins=["*"]
# combination let any origin make "credentialed" requests, since browsers
# reject a literal wildcard alongside credentials and Starlette works around
# that by reflecting the request's Origin header instead). allow_origins is
# configurable per-environment instead of "*"; defaults cover local frontend dev.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "SAFEDERM_ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")
    if origin.strip()
]

# ---------------------------------------------------------------------------
# App state -- populated at startup (see lifespan below), read by handlers
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: Optional[nn.Module] = None
transform = None
checkpoint_loaded = False
model_supports_mc_dropout = False
calibration: Optional[CalibrationArtifact] = None

# The model's .training flag gets toggled per-request by enable_mc_dropout()
# (see src/calibration.py). One shared model instance means two concurrent
# requests dispatched to different threadpool threads could race on that
# flag. Serializing inference with a lock is the simple, correct fix at
# this stage -- it trades away concurrent *throughput* for correctness.
# A real scale-up (multiple model replicas / a dedicated inference
# service) is a later problem, not one to leave a race condition for now.
_inference_lock = threading.Lock()


def _model_has_dropout(m: nn.Module) -> bool:
    return any(isinstance(module, nn.Dropout) for module in m.modules())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook. Replaces the deprecated @app.on_event("startup")."""
    global model, transform, checkpoint_loaded, model_supports_mc_dropout, calibration

    entry = MODEL_REGISTRY[MODEL_VARIANT]
    model = entry["build_fn"]()
    checkpoint_path = entry["checkpoint_path"]

    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        checkpoint_loaded = True
        logger.info("Loaded %s checkpoint from %s", MODEL_VARIANT, checkpoint_path)
    else:
        checkpoint_loaded = False
        logger.warning(
            "No checkpoint found at %s -- serving %s with UNTRAINED weights. "
            "Predictions will be meaningless until this checkpoint exists.",
            checkpoint_path, MODEL_VARIANT,
        )

    model.to(device)
    model.eval()
    transform = get_eval_transforms()
    model_supports_mc_dropout = _model_has_dropout(model)

    calibration = CalibrationArtifact.load_if_exists(CALIBRATION_ARTIFACT_PATH)
    if calibration is None:
        logger.warning(
            "No calibration artifact at %s -- serving raw softmax confidence "
            "(calibrated=false in every response) until 08_calibration_conformal.ipynb "
            "produces one.", CALIBRATION_ARTIFACT_PATH,
        )
    elif not model_supports_mc_dropout:
        logger.warning(
            "Calibration artifact found but the '%s' model has no Dropout "
            "layers -- MC Dropout uncertainty needs one (see src/model.py's "
            "_TransformerHead). Serving raw softmax confidence instead.",
            MODEL_VARIANT,
        )
    else:
        logger.info(
            "Calibration artifact loaded: T=%.3f, entropy_threshold=%.3f, "
            "confidence_threshold=%.3f -- serving calibrated, tiered predictions.",
            calibration.temperature, calibration.entropy_threshold, calibration.confidence_threshold,
        )

    yield
    # No shutdown work needed -- nothing external held open (no DB/file handles).


app = FastAPI(
    title="SafeDerm API",
    description="AI-assisted skin lesion triage system API.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    diagnosis: str
    risk_group: str
    tier: str  # "normal" / "concerning" / "uncertain" once calibrated, else "not_calibrated"
    confidence: float  # calibrated MC-Dropout confidence if `calibrated` else raw softmax
    calibrated: bool  # False -> `confidence`/`tier` are not backed by 08's calibration yet
    probabilities: Dict[str, float]
    model_variant: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool        # model object constructed successfully
    checkpoint_loaded: bool   # trained weights actually loaded -- NOT the same as
                               # model_loaded: a model builds fine with random
                               # weights even if the checkpoint file is missing
    model_variant: str
    calibration_loaded: bool


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _run_inference(tensor: torch.Tensor) -> dict:
    """The actual (synchronous, CPU-bound) forward-pass work, called via
    run_in_threadpool from the /predict handler so MC Dropout's ~20
    forward passes don't block the event loop for other concurrent
    requests -- the original single-pass version had this problem too,
    just 20x less of it.
    """
    with _inference_lock:
        if calibration is not None and model_supports_mc_dropout:
            mean_probs, entropy = mc_dropout_predict(
                model, tensor, n_passes=calibration.mc_dropout_passes, temperature=calibration.temperature
            )
            probs = mean_probs[0]
            predicted_idx = int(probs.argmax().item())
            dx = ALL_CLASSES[predicted_idx]
            risk = risk_group(dx)
            tier = assign_tier(
                predicted_risk_group=risk,
                predictive_entropy=float(entropy[0].item()),
                calibrated_confidence=float(probs[predicted_idx].item()),
                entropy_threshold=calibration.entropy_threshold,
                confidence_threshold=calibration.confidence_threshold,
            )
            calibrated = True
        else:
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits[0], dim=0)
            predicted_idx = int(probs.argmax().item())
            dx = ALL_CLASSES[predicted_idx]
            risk = risk_group(dx)
            tier = "not_calibrated"
            calibrated = False

        return {
            "diagnosis": dx,
            "risk_group": risk,
            "tier": tier,
            "confidence": float(probs[predicted_idx].item()),
            "calibrated": calibrated,
            "probabilities": {ALL_CLASSES[i]: float(probs[i].item()) for i in range(len(ALL_CLASSES))},
            "model_variant": MODEL_VARIANT,
        }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if model is None or transform is None:
        logger.error("Attempted prediction but model is not loaded.")
        raise HTTPException(status_code=500, detail="Model not loaded.")

    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning("Invalid file type uploaded: %s", file.content_type)
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        logger.warning("Uploaded file exceeds 5MB limit.")
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        return await run_in_threadpool(_run_inference, tensor)
    except HTTPException:
        raise
    except Exception:
        # Full traceback goes to the server log; the client gets a message
        # that can't leak internals (stack traces, file paths, library
        # versions). The original version returned str(e) straight to the
        # caller.
        logger.exception("Unexpected error while processing a prediction request")
        raise HTTPException(
            status_code=500,
            detail="Internal error while processing the image. Please try again.",
        )


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        checkpoint_loaded=checkpoint_loaded,
        model_variant=MODEL_VARIANT,
        calibration_loaded=calibration is not None and model_supports_mc_dropout,
    )
