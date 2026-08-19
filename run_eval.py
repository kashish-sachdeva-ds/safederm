import sys
sys.path.append('e:\\New folder (2)\\safederm')
from pathlib import Path
from src.gateway import load_gateway_model, verify_image_is_skin
from src.config import IMAGES_DIR, TRAIN_SPLIT_PATH
from src.dataset import SkinLesionDataset

gateway = load_gateway_model()
train_dataset = SkinLesionDataset(TRAIN_SPLIT_PATH, IMAGES_DIR, transform=None)
skin_image_paths = [train_dataset.get_image_path(i) for i in range(20)]

OOD_SAMPLES_DIR = Path('e:\\New folder (2)\\safederm\\notebooks\\gateway_ood_samples')
non_skin_image_paths = sorted(OOD_SAMPLES_DIR.glob('*.*'))

skin_scores = [verify_image_is_skin(str(p), gateway)[1] for p in skin_image_paths]
non_skin_scores = [verify_image_is_skin(str(p), gateway)[1] for p in non_skin_image_paths]

print(f'Skin min: {min(skin_scores):.3f}')
print(f'Skin max: {max(skin_scores):.3f}')
print(f'Non-skin min: {min(non_skin_scores):.3f}')
print(f'Non-skin max: {max(non_skin_scores):.3f}')
