"""Project-wide path configuration for SafeDerm.

Single source of truth for every file path used across notebooks and src/.
Import from here instead of hardcoding paths -- keeps every notebook working
regardless of who runs it or where the repo is cloned.
"""

from pathlib import Path

# Repo root -- two levels up from this file (src/config.py -> src/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Kaggle dataset slug -- used by the download step
KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"

# Raw data layout (after extraction/consolidation)
IMAGES_DIR = RAW_DATA_DIR / "images"  # all 10,015 jpgs, merged into one folder
METADATA_PATH = RAW_DATA_DIR / "HAM10000_metadata.csv"

# Processed splits (written once the lesion-level split notebook runs)
TRAIN_SPLIT_PATH = PROCESSED_DATA_DIR / "train_split.csv"
VAL_SPLIT_PATH = PROCESSED_DATA_DIR / "val_split.csv"
TEST_SPLIT_PATH = PROCESSED_DATA_DIR / "test_split.csv"

# Model checkpoints
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_resnet50.pt"
CHAMPION_MODEL_PATH = MODELS_DIR / "champion_transformer.pt"

# Calibration artifact (temperature + tier thresholds), written by
# 08_calibration_conformal.ipynb, read by api/main.py at startup.
CALIBRATION_ARTIFACT_PATH = MODELS_DIR / "calibration.json"

# Small metrics summaries (val_loss/val_accuracy/malignant_recall), written
# alongside each checkpoint so 07_champion_model.ipynb can load 06's result
# and print a direct comparison table instead of requiring someone to
# eyeball two separate notebooks' printed output.
BASELINE_METRICS_PATH = MODELS_DIR / "baseline_metrics.json"
CHAMPION_METRICS_PATH = MODELS_DIR / "champion_metrics.json"

# Sanity-check constants used by 01_data_extraction.ipynb
EXPECTED_IMAGE_COUNT = 10015
EXPECTED_CLASSES = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
