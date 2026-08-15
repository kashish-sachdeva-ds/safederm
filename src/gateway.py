"""
Input Safety Gateway (OOD Detection)
-------------------------------------
Uses a pre-trained BiomedCLIP model (zero-shot, no fine-tuning) to verify that an
uploaded image plausibly contains human skin before it is forwarded to the
clinical SafeDerm ResNet-50 model.

Design notes:
- The CLIP model/processor are expensive to load, so this module exposes a
  `load_gateway_model()` function meant to be called ONCE at app startup
  (see api/main.py lifespan integration) and reused across requests.
- `verify_image_is_skin()` takes an already-loaded model/processor so the
  route handler doesn't pay the load cost per-request.
- Threshold is configurable, not hardcoded, so Member 3 can tune sensitivity
  after looking at real validation data instead of guessing 50%.
"""

from dataclasses import dataclass
from io import BytesIO
import typing

import torch
import open_clip
from PIL import Image

CLIP_MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# Multiple phrasings per class improve zero-shot robustness vs. a single
# prompt — CLIP's score can be sensitive to exact wording.
SKIN_PROMPTS = [
    "a close-up medical photograph of human skin with a mole, lesion, or rash",
    "a dermatology photo of a skin condition",
    "a close-up photo of a patch of human skin",
]
NON_SKIN_PROMPTS = [
    "a photograph of a random object, scenery, animal, or person",
    "a photo of an everyday object or landscape",
    "a picture unrelated to skin or dermatology",
]

# Anything below this on the skin-side aggregate score is rejected.
DEFAULT_SKIN_THRESHOLD = 0.5


@dataclass
class GatewayModel:
    model: torch.nn.Module
    preprocess: typing.Callable
    tokenizer: typing.Callable
    device: str


def load_gateway_model(device: str | None = None) -> GatewayModel:
    """Load CLIP once at startup. Call this from the FastAPI lifespan, not
    per-request — loading takes real time and memory."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model = model.to(device)
    model.eval()
    return GatewayModel(model=model, preprocess=preprocess, tokenizer=tokenizer, device=device)


def _to_pil(image) -> Image.Image:
    """Accept raw bytes, a file-like object, or an already-decoded PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(BytesIO(image)).convert("RGB")
    return Image.open(image).convert("RGB")


@torch.no_grad()
def verify_image_is_skin(
    image,
    gateway: GatewayModel,
    threshold: float = DEFAULT_SKIN_THRESHOLD,
) -> tuple[bool, float]:
    """
    Returns (is_skin, skin_score).

    is_skin: whether the aggregate skin-prompt probability clears `threshold`.
    skin_score: the raw aggregate probability, useful for logging/debugging
                and for tuning the threshold later.
    """
    pil_image = _to_pil(image)
    prompts = SKIN_PROMPTS + NON_SKIN_PROMPTS
    n_skin = len(SKIN_PROMPTS)

    image_tensor = gateway.preprocess(pil_image).unsqueeze(0).to(gateway.device)
    text_tokens = gateway.tokenizer(prompts).to(gateway.device)

    image_features, text_features, logit_scale = gateway.model(image_tensor, text_tokens)
    logits = (logit_scale * image_features @ text_features.T)
    
    # (1, n_prompts) similarity logits -> softmax across ALL prompts together,
    # then sum the skin-side probabilities. This is more stable than running
    # a separate binary softmax per prompt pair.
    probs = logits.softmax(dim=-1)[0]
    skin_score = probs[:n_skin].sum().item()

    return skin_score >= threshold, skin_score
