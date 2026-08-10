import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch

from src.model import build_baseline_model
from src.config import BASELINE_MODEL_PATH
from src.transforms import get_eval_transforms
from src.labels import ALL_CLASSES, risk_group

app = FastAPI(
    title="SafeDerm API",
    description="AI-assisted skin lesion triage system API.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = None

@app.on_event("startup")
def load_model():
    global model, transform
    try:
        model = build_baseline_model()
        if BASELINE_MODEL_PATH.exists():
            state_dict = torch.load(BASELINE_MODEL_PATH, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"Loaded model weights from {BASELINE_MODEL_PATH}")
        else:
            print(f"Warning: Model weights not found at {BASELINE_MODEL_PATH}. Using untrained weights.")
        
        model.to(device)
        model.eval()
        transform = get_eval_transforms()
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or transform is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            
        max_prob, predicted_idx = torch.max(probs, 0)
        dx = ALL_CLASSES[predicted_idx.item()]
        risk = risk_group(dx)
        
        class_probs = {ALL_CLASSES[i]: float(probs[i].item()) for i in range(len(ALL_CLASSES))}
        
        return {
            "diagnosis": dx,
            "risk_group": risk,
            "confidence": float(max_prob.item()),
            "probabilities": class_probs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
