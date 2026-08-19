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

skin_scores = [(str(p), verify_image_is_skin(str(p), gateway)[1]) for p in skin_image_paths]
non_skin_scores = [(str(p), verify_image_is_skin(str(p), gateway)[1]) for p in non_skin_image_paths]

skin_scores.sort(key=lambda x: x[1])
non_skin_scores.sort(key=lambda x: x[1], reverse=True)

print('--- Lowest scoring skin images ---')
for p, s in skin_scores[:5]: print(f'{s:.3f}: {Path(p).name}')

print('\\n--- Highest scoring non-skin images ---')
for p, s in non_skin_scores[:5]: print(f'{s:.3f}: {Path(p).name}')
