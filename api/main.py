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
<<<<<<< HEAD
from typing import Any, Dict, Optional
=======
from typing import Dict, Optional, cast
>>>>>>> ce1e3aa (resolving conflict)
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms as tv_transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.calibration import CalibrationArtifact, assign_tier, mc_dropout_predict
from src.config import BASELINE_MODEL_PATH, CALIBRATION_ARTIFACT_PATH, CHAMPION_MODEL_PATH, GATEWAY_THRESHOLD
from src.labels import ALL_CLASSES, risk_group
from src.model import build_baseline_model, build_champion_model
from src.transforms import get_eval_transforms
from src.gateway import load_gateway_model, verify_image_is_skin
from src.near_ood import (
    load_feature_bank_if_exists,
    verify_in_distribution,
    DEFAULT_DISTANCE_THRESHOLD,
    resnet_avgpool_embedding_fn,
    champion_embedding_fn,
    FeatureBank,
)

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
transform: Optional[tv_transforms.Compose] = None
checkpoint_loaded = False
model_supports_mc_dropout = False
calibration: Optional[CalibrationArtifact] = None
gw_model = None
feature_bank: Optional[FeatureBank] = None

_inference_lock = threading.Lock()


def _model_has_dropout(m: nn.Module) -> bool:
    return any(isinstance(module, nn.Dropout) for module in m.modules())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, transform, checkpoint_loaded, model_supports_mc_dropout, calibration
    global gw_model, feature_bank

    entry = MODEL_REGISTRY[MODEL_VARIANT]
    # Local, non-Optional handle -- avoids re-narrowing the Optional[nn.Module]
    # global on every line below; assigned to the global once at the end.
    built_model: nn.Module = entry["build_fn"]()
    checkpoint_path = entry["checkpoint_path"]

    if checkpoint_path.exists():
        assert model is not None
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        built_model.load_state_dict(state_dict)
        checkpoint_loaded = True
        logger.info("Loaded %s checkpoint from %s", MODEL_VARIANT, checkpoint_path)
    else:
        checkpoint_loaded = False
        logger.warning(
            "No checkpoint found at %s -- serving %s with UNTRAINED weights. "
            "Predictions will be meaningless until this checkpoint exists.",
            checkpoint_path, MODEL_VARIANT,
        )

<<<<<<< HEAD
    assert model is not None
    model.to(device)
    model.eval()
=======
    built_model.to(device)
    built_model.eval()
    model = built_model
>>>>>>> ce1e3aa (resolving conflict)
    transform = get_eval_transforms()
    model_supports_mc_dropout = _model_has_dropout(built_model)

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

    logger.info("Loading Input Safety Gateway model...")
    gw_model = load_gateway_model(str(device))

    logger.info("Loading Feature Bank (Gateway 2)...")
<<<<<<< HEAD
    try:
        feature_bank = load_feature_bank(Path("models"), str(device))
    except FileNotFoundError:
        logger.warning("feature_bank.pt not found. Gateway 2 will be disabled.")
        feature_bank = None
=======
    feature_bank = load_feature_bank_if_exists(Path("models"), str(device), MODEL_VARIANT)
    if feature_bank is None:
        logger.warning(
            "No feature bank found at models/feature_bank_%s.pt -- Gateway 2 "
            "(near-OOD detection) will be SKIPPED at request time "
            "(gateway2_checked=false in every response) until "
            "notebooks/near_ood_feature_bank.ipynb builds one. Gateway 1 "
            "(BiomedCLIP skin/non-skin check) still runs.",
            MODEL_VARIANT,
        )
>>>>>>> ce1e3aa (resolving conflict)

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
    model_loaded: bool
    checkpoint_loaded: bool
    model_variant: str
    calibration_loaded: bool
    feature_bank_loaded: bool  # False -> Gateway 2 (near-OOD) is skipped at /predict time, not silently passed

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


<<<<<<< HEAD
def _run_inference(tensor: torch.Tensor) -> Dict[str, Any]:
=======
def _run_inference(tensor: torch.Tensor) -> dict:
    # predict() already guards model is None before dispatching here, but
    # pyright checks this threadpool-called helper independently and can't
    # see that caller guarantee across the call boundary.
>>>>>>> ce1e3aa (resolving conflict)
    assert model is not None
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


def _check_gateway1(image: Image.Image) -> tuple[bool, float]:
    assert gw_model is not None
    return verify_image_is_skin(image, gateway=gw_model, threshold=GATEWAY_THRESHOLD)


def _check_gateway2(tensor: torch.Tensor) -> tuple[bool, float]:
    assert model is not None
    assert feature_bank is not None
    emb_fn = resnet_avgpool_embedding_fn if MODEL_VARIANT == "baseline" else champion_embedding_fn
    return verify_in_distribution(
        tensor,
        model,
        feature_bank,
        threshold=DEFAULT_DISTANCE_THRESHOLD,
        embedding_fn=emb_fn,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if model is None or transform is None or gw_model is None:
        logger.error("Attempted prediction but models are not loaded.")
        raise HTTPException(status_code=500, detail="Models not loaded.")

    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning("Invalid file type uploaded: %s", file.content_type)
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        logger.warning("Uploaded file exceeds 5MB limit.")
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
<<<<<<< HEAD
        
        # Gateway Check 1 (Far-OOD): Is it skin?
        is_skin, skin_score = await run_in_threadpool(
            verify_image_is_skin,
            image, 
            gateway=gw_model, 
            threshold=GATEWAY_THRESHOLD,
        )
        if not is_skin:
            logger.warning("Gateway 1 check failed. Image is likely out-of-distribution (non-skin).")
            raise HTTPException(
                status_code=400, 
                detail="Invalid image detected. Please upload a clear medical photograph of human skin."
            )

        assert transform is not None
        tensor = transform(image).unsqueeze(0).to(device)  # type: ignore
        
        # Gateway Check 2 (Near-OOD): Is it dermoscopy quality?
        if feature_bank is not None:
            emb_fn = resnet_avgpool_embedding_fn if MODEL_VARIANT == "baseline" else champion_embedding_fn
            is_in_dist, distance = await run_in_threadpool(
                verify_in_distribution,
                tensor,
                model,
                feature_bank,
                threshold=DEFAULT_DISTANCE_THRESHOLD,
                embedding_fn=emb_fn
=======

        # Gateway Check 1 (Far-OOD): Is it skin? Model call, so it belongs
        # in the threadpool same as inference -- it was blocking the event
        # loop before.
        is_skin, skin_score = await run_in_threadpool(_check_gateway1, image)
        if not is_skin:
            logger.warning("Gateway 1 check failed (skin_score=%.3f).", skin_score)
            raise HTTPException(
                status_code=400,
                detail="Invalid image detected. Please upload a clear medical photograph of human skin.",
>>>>>>> ce1e3aa (resolving conflict)
            )
            if not is_in_dist:
                logger.warning(f"Gateway 2 check failed. Image is skin but not in-distribution (distance: {distance:.3f}).")
                raise HTTPException(
                    status_code=400,
                    detail="Image detected as skin but does not match clinical quality (e.g. poor lighting, non-dermoscopic). Please upload a clear dermoscopy image."
                )

        assert transform is not None  # guarded by the is-None check above; pyright can't see across the await
        # Compose.__call__ is typed generically (returns the same type as
        # its input) in torchvision's stubs, since it can't statically know
        # a ToTensor step is inside -- cast() reflects the real runtime type.
        tensor = cast(torch.Tensor, transform(image)).unsqueeze(0).to(device)

        # Gateway Check 2 (Near-OOD): Is it dermoscopy quality? Skipped
        # gracefully (not silently passed) if no feature bank has been
        # built yet for this model variant.
        if feature_bank is not None:
            is_in_dist, distance = await run_in_threadpool(_check_gateway2, tensor)
            if not is_in_dist:
                # Distance is logged server-side only -- it's an internal
                # detection-threshold detail, not something a client needs
                # (or should be able to use to probe the threshold).
                logger.warning("Gateway 2 check failed (distance=%.3f).", distance)
                raise HTTPException(
                    status_code=400,
                    detail="Image detected as skin but does not match clinical quality "
                    "(e.g. poor lighting, non-dermoscopic). Please upload a clear dermoscopy image.",
                )

        return await run_in_threadpool(_run_inference, tensor)
    except HTTPException:
        raise
    except Exception:
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
        feature_bank_loaded=feature_bank is not None,
    )
