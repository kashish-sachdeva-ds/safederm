# SafeDerm

AI-assisted skin lesion triage system. Classifies dermatoscopic images into 7 diagnostic categories and flags potentially malignant lesions for dermatologist review, backed by calibrated confidence scores rather than raw model output.

## ⚠️ Scope & Safety

This tool screens for **7 specific dermoscopic lesion types** (below). It is **not** a general-purpose skin diagnosis tool, is not designed to identify other skin conditions (eczema, acne, psoriasis, burns, etc.), and is not a substitute for professional medical evaluation. Always consult a dermatologist for any skin concern.

## Overview

Skin cancer screening is time-intensive and dermatologist access is limited. SafeDerm classifies a dermatoscopic image into one of 7 diagnoses, groups the result into a malignant/benign risk category, and — once calibration is complete — routes the result through a 3-tier confidence system rather than presenting a single raw probability. Full business framing in `docs/decisions/ADR-001`.

## Dataset

[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) ("Human Against Machine with 10000 training images"). 10,015 dermatoscopic images, 7 diagnostic classes:

| Risk group | Classes |
|---|---|
| **Malignant** (needs referral) | melanoma (`mel`), basal cell carcinoma (`bcc`), actinic keratosis (`akiec` — precancerous) |
| **Benign** | melanocytic nevi (`nv`), benign keratosis (`bkl`), dermatofibroma (`df`), vascular lesions (`vasc`) |

Grouping rationale — including why precancerous `akiec` is treated as malignant — in `docs/decisions/ADR-004`.

Split is **lesion-level, not image-level**, stratified by class, to prevent the same lesion's photos from leaking across train/val/test. See `docs/decisions/ADR-003`.

## Project Structure

```
SafeDerm/
├── data/
│   ├── raw/              # HAM10000 images + metadata (gitignored — see Setup)
│   └── processed/        # train/val/test split CSVs
├── docs/
│   └── decisions/        # ADRs — architectural decision records
├── models/                # trained checkpoints (gitignored — see Releases)
├── notebooks/              # numbered, phase-gated pipeline
├── src/                    # reusable code — single source of truth
│   ├── config.py           # paths and constants
│   ├── labels.py           # class definitions, malignant/benign grouping
│   ├── transforms.py       # image preprocessing/augmentation
│   ├── dataset.py          # PyTorch Dataset
│   ├── model.py            # model architectures
│   └── engine.py           # training/eval loops
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Setup

1. Clone the repo, create and activate a virtual environment.
2. `pip install -r requirements.txt`
3. `pip install -e .` — installs the project itself so `from src...` imports work from any notebook.
4. Get the dataset — see `notebooks/01_data_extraction.ipynb`. Supports both Kaggle CLI download (needs `~/.kaggle/kaggle.json`) and manual download; either way, drop the zip in `data/raw/` and the notebook handles extraction.
5. (Optional, if not training from scratch) Download a trained checkpoint from the repo's [Releases](../../releases) page and place it in `models/`.

## Running the Pipeline

Notebooks are numbered and phase-gated — each depends on the previous one's output, run in order:

| # | Notebook | Purpose | Status |
|---|---|---|---|
| 01 | `01_data_extraction.ipynb` | Download, extract, consolidate HAM10000 | ✅ Done |
| 02 | `02_data_understanding.ipynb` | Inspect metadata, lesion-level train/val/test split | ✅ Done |
| 03 | `03_eda.ipynb` | Class balance, demographics, sample images (train split only) | ✅ Done |
| 04 | `04_feature_engineering.ipynb` | Image preprocessing/augmentation pipeline | ✅ Done |
| 05 | `05_pipeline.ipynb` | PyTorch Dataset/DataLoader, batch sanity checks | ✅ Done |
| 06 | `06_baseline_model.ipynb` | ResNet-50 baseline, weighted loss | ✅ Done |
| 07 | `07_champion_model.ipynb` | CNN + Transformer architecture | 🔲 Planned |
| 08 | `08_calibration_conformal.ipynb` | Temperature scaling, ECE/MCE/Brier, 3-tier thresholds | 🔲 Planned |

## Current Results — Baseline (ResNet-50)

| Metric | Value |
|---|---|
| Validation accuracy | 77.4% |
| Malignant recall | 83.1% (255/308 correctly flagged) |
| Missed malignant cases | 52 (17%) |

Full per-class report and confusion matrix in `06_baseline_model.ipynb`. This is the number `07`'s architecture needs to beat — specifically on malignant recall, not just overall accuracy. Class imbalance handling (weighted loss) documented in `docs/decisions/ADR-005`.

## Architecture Decisions

Every significant design decision is recorded in `docs/decisions/`:

- **ADR-001** — Business problem framing
- **ADR-002** — Technical architecture
- **ADR-003** — Dataset split strategy (lesion-level, stratified)
- **ADR-004** — Malignant/benign risk grouping
- **ADR-005** — Class imbalance handling (weighted loss)

## Roadmap

- [x] Data pipeline (extraction → split → EDA → preprocessing)
- [x] Baseline model
- [ ] Champion model (CNN + Transformer)
- [ ] Calibration + 3-tier confidence system
- [ ] Out-of-distribution / input safety testing
- [ ] FastAPI backend + Docker deployment
- [ ] React frontend
- [ ] Cost-sensitive business case, demo, final report

## Team

4-person project. Roles: ML Engineer (data pipeline, modeling, calibration), MLOps/Data, Backend Engineer, Frontend/Product. Full breakdown in the team's shared Notion workspace.

## License

See `LICENSE`.
