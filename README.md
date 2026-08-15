# SafeDerm

AI-assisted skin lesion triage system. Classifies dermatoscopic images into 7 diagnostic categories and flags potentially malignant lesions for dermatologist review, backed by calibrated confidence scores rather than raw model output.

## ⚠️ Scope & Safety

This tool screens for **7 specific dermoscopic lesion types** (below). It is **not** a general-purpose skin diagnosis tool, is not designed to identify other skin conditions (eczema, acne, psoriasis, burns, etc.), and is not a substitute for professional medical evaluation. Always consult a dermatologist for any skin concern.

## Overview

Skin cancer screening is time-intensive and dermatologist access is limited. SafeDerm classifies a dermatoscopic image into one of 7 diagnoses, groups the result into a malignant/benign risk category, and routes the result through a 3-tier confidence system (`normal` / `concerning` / `uncertain`) rather than presenting a single raw probability — see `docs/decisions/ADR-002`. Full business framing in `docs/decisions/ADR-001`.

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
├── api/                     # FastAPI serving layer
│   ├── main.py               # /predict, /health
│   └── test_main.py
├── frontend/                # React (Vite) upload + result UI
│   └── src/
├── data/
│   ├── raw/                  # HAM10000 images + metadata (gitignored — see Setup)
│   └── processed/            # train/val/test split CSVs
├── docs/
│   └── decisions/            # ADRs — architectural decision records
├── models/                   # trained checkpoints + calibration artifact (gitignored — see Releases)
├── notebooks/                 # numbered, phase-gated pipeline
├── src/                        # reusable code — single source of truth
│   ├── config.py                # paths and constants
│   ├── labels.py                 # class definitions, malignant/benign grouping
│   ├── transforms.py              # image preprocessing/augmentation
│   ├── dataset.py                  # PyTorch Dataset
│   ├── model.py                     # model architectures (baseline + champion)
│   ├── engine.py                     # training/eval loops, shared metrics
│   └── calibration.py                 # temperature scaling, MC Dropout, 3-tier routing
├── .github/workflows/ci.yml    # runs api/test_main.py + frontend build on every push/PR
├── pyproject.toml
├── requirements.txt
├── requirements-gpu.txt        # optional, for local CUDA training outside Colab/Kaggle
└── requirements-dev.txt
```

## Setup

**Backend / modeling:**
1. Clone the repo, create and activate a virtual environment.
2. `pip install -r requirements.txt` (CPU by default; see `requirements-gpu.txt` for local NVIDIA GPU training).
3. `pip install -e .` — installs the project itself so `from src...` imports work from any notebook.
4. Get the dataset — see `notebooks/01_data_extraction.ipynb`. Supports both Kaggle CLI download (needs `~/.kaggle/kaggle.json`) and manual download; either way, drop the zip in `data/raw/` and the notebook handles extraction.
5. (Optional, if not training from scratch) Download a trained checkpoint from the repo's [Releases](../../releases) page and place it in `models/`.

**API:**
```
cd api && pip install -r requirements.txt   # self-sufficient on its own — pulls in the root requirements.txt too
uvicorn api.main:app --reload
```
Configurable via env vars: `SAFEDERM_MODEL_VARIANT` (`baseline` | `champion`, default `baseline`), `SAFEDERM_ALLOWED_ORIGINS` (comma-separated, default covers local frontend dev). Or `docker compose up` to run the whole thing containerized.

**Frontend:**
```
cd frontend && npm install && npm run dev
```
Expects the API at `http://localhost:8000` by default — copy `.env.example` to `.env.local` to point it elsewhere.

## Running the Pipeline

Notebooks are numbered and phase-gated — each depends on the previous one's output, run in order:

| # | Notebook | Purpose | Status |
|---|---|---|---|
| 01 | `01_data_extraction.ipynb` | Download, extract, consolidate HAM10000 | ✅ Done |
| 02 | `02_data_understanding.ipynb` | Inspect metadata, lesion-level train/val/test split | ✅ Done |
| 03 | `03_eda.ipynb` | Class balance, demographics, sample images (train split only) | ✅ Done |
| 04 | `04_feature_engineering.ipynb` | Image preprocessing/augmentation pipeline | ✅ Done |
| 05 | `05_pipeline.ipynb` | PyTorch Dataset/DataLoader, batch sanity checks | ✅ Done |
| 06 | `06_baseline_model.ipynb` | ResNet-50 baseline, weighted loss, two-phase recipe (ADR-006) | 🔶 Recipe updated — needs a fresh run |
| 07 | `07_champion_model.ipynb` | CNN + Transformer architecture | 🔶 Code complete — needs a training run |
| 08 | `08_calibration_conformal.ipynb` | Temperature scaling, MC Dropout, ECE/MCE/Brier, 3-tier thresholds, OOD stress test | 🔶 Code complete — needs `07`'s checkpoint first |

🔶 = code written and tested (unit + integration tests against synthetic data), but not yet run against the real dataset — the difference between "the pipeline works" and "we have real numbers from it" matters, so this status isn't collapsed into ✅ until it's actually been run.

## Current Results — Baseline (ResNet-50)

| Metric | Value |
|---|---|
| Validation accuracy | 77.4% |
| Malignant recall | 83.1% (255/308 correctly flagged) |
| Missed malignant cases | 52 (17%) |

**These numbers are from the original single-LR, fixed-10-epoch recipe** — the training curve behind them shows clear overfitting past epoch 5 (see ADR-006). `06` has since been updated to a two-phase warmup + discriminative-LR + early-stopping recipe; re-running it will produce a new (and more honestly-trained) number to replace this table with. Full per-class report and confusion matrix in `06_baseline_model.ipynb`. This is the number `07`'s architecture needs to beat — specifically on malignant recall, not just overall accuracy. Class imbalance handling (weighted loss) documented in `docs/decisions/ADR-005`.

## Architecture Decisions

Every significant design decision is recorded in `docs/decisions/`:

- **ADR-001** — Business problem framing
- **ADR-002** — Technical architecture
- **ADR-003** — Dataset split strategy (lesion-level, stratified)
- **ADR-004** — Malignant/benign risk grouping
- **ADR-005** — Class imbalance handling (weighted loss)
- **ADR-006** — Training recipe fix and champion model architecture

## Roadmap

- [x] Data pipeline (extraction → split → EDA → preprocessing)
- [x] Baseline model (recipe updated per ADR-006 — re-run pending)
- [x] CI (API tests + frontend build run automatically on push/PR)
- [ ] Champion model (CNN + Transformer) — code complete, training run pending
- [ ] Calibration + 3-tier confidence system — code complete, pending `07`'s checkpoint
- [ ] Out-of-distribution / input safety testing — stress-test built into `08`, pending a real run
- [x] FastAPI backend + Docker deployment
- [ ] React frontend — scaffolded (upload + tiered result display), pending real backend results to design against
- [ ] Cost-sensitive business case, demo, final report

## Team

4-person project. Roles: ML Engineer (data pipeline, modeling, calibration), MLOps/Data, Backend Engineer, Frontend/Product. Full breakdown in the team's shared Notion workspace.

## License

See `LICENSE`.
