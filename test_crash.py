import sys
sys.path.append("e:\\New folder (2)\\safederm")
import matplotlib.pyplot as plt
from pathlib import Path
from src.gateway import load_gateway_model, verify_image_is_skin
from src.config import IMAGES_DIR, TRAIN_SPLIT_PATH
from src.dataset import SkinLesionDataset

print("Loading gateway...")
gateway = load_gateway_model()
print(f"Gateway loaded on {gateway.device}")

print("Loading dataset...")
N_SAMPLES = 20
train_dataset = SkinLesionDataset(TRAIN_SPLIT_PATH, IMAGES_DIR, transform=None)
skin_image_paths = [train_dataset.get_image_path(i) for i in range(N_SAMPLES)]
print(f"Collected {len(skin_image_paths)} real skin images")

OOD_SAMPLES_DIR = Path("notebooks/gateway_ood_samples")
non_skin_image_paths = sorted(OOD_SAMPLES_DIR.glob("*.*"))
print(f"Collected {len(non_skin_image_paths)} non-skin images")

print("Running inference...")
skin_scores = [verify_image_is_skin(str(p), gateway)[1] for p in skin_image_paths]
non_skin_scores = [verify_image_is_skin(str(p), gateway)[1] for p in non_skin_image_paths]
print("Inference completed!")
