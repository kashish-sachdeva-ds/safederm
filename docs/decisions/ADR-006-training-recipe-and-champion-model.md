# ADR-006: Training Recipe Fix and Champion Model Architecture

## Status
Accepted

## Context
The baseline (`06_baseline_model.ipynb`, ADR-002/ADR-005) fully fine-tuned
all 25M ResNet-50 parameters at a single LR (`Adam`, `lr=1e-4`), for a
fixed 10 epochs, with no weight decay, no LR schedule, and no seeding.
The training curve shows the result: `train_loss` fell monotonically
(1.28 → 0.22) while `val_loss` bottomed out at epoch 5 (0.6434) and never
beat that again through epoch 10 (0.68–0.87) — a clear overfitting
signature. Epochs 6–10 were spent training on a model that was already
past its best validation performance.

This mattered beyond the baseline itself: `07_champion_model.ipynb` was
always going to share `src/engine.py`'s training loop with `06` "so any
difference in results reflects the architecture, not a difference in how
each was trained." Handing both models the same *flawed* recipe would
have kept that comparison technically fair but pointlessly so — neither
number would represent what either architecture could actually do, and a
heavier champion model (more parameters, same ~7K training images) would
likely overfit harder under the same recipe, not less.

Separately, ADR-002 commits to MC Dropout for uncertainty estimation
(`08_calibration_conformal.ipynb`). MC Dropout needs the model to contain
an actual `nn.Dropout` layer to sample stochastically over at inference.
`build_baseline_model()` — a plain `torchvision.models.resnet50` with the
final FC layer swapped — has none. This had to be resolved as part of
the champion model's architecture, not left for `08` to discover.

## Decision

**Training recipe** (`src/engine.py`, used by both `06` and `07`):
- `set_seed(42)` — seeds python/numpy/torch so a run is reproducible, not
  just the data split (ADR-003 only covered the latter).
- Two-phase fine-tuning: phase 1 freezes the pretrained backbone and
  trains only the head for `WARMUP_EPOCHS` (2); phase 2 unfreezes
  everything with discriminative learning rates — `HEAD_LR=1e-3` for the
  freshly-initialized head, a much smaller `BACKBONE_LR=1e-5` for the
  pretrained layers (`get_param_groups()`, `freeze_backbone()`,
  `unfreeze_all()`).
- `AdamW` with `WEIGHT_DECAY=1e-4` replaces plain `Adam` with no
  regularization.
- `ReduceLROnPlateau` drops the LR when val_loss stalls, instead of
  holding a constant rate for the whole run.
- `EarlyStopper(patience=4)` stops a run once val_loss hasn't improved
  in 4 epochs, instead of always spending a fixed epoch budget.
- Checkpoint selection stays on weighted val_loss by default (ADR-005),
  but `fit()` accepts `selection_metric="malignant_recall"` as an
  alternative — the project's actual safety metric, not just a proxy for
  it. `malignant_recall()` itself moved from an inline calculation in
  `06`'s cell 16 into `src/engine.py`, since `07` (and `08`'s threshold
  fitting) need the identical calculation, not a re-copy of it.

**Champion model architecture** (`src/model.py::ChampionModel`):
A ResNet-50 backbone through its last conv block (`conv1` … `layer4`,
everything except the original `avgpool`/`fc`) feeds a small Transformer
encoder over the resulting 7×7=49 spatial grid, via a learnable CLS
token and positional embeddings, instead of average-pooling straight
into a linear layer. `d_model=256`, 8 attention heads, 3 encoder layers.
Rationale from ADR-002: self-attention over the spatial grid produces an
attention map that's directly visualizable — `attention_rollout()`
returns the CLS token's attention distribution over the 49 patches — a
more literal "where did the model look" explanation than Grad-CAM
approximates after the fact on a plain CNN.

The classifier head (`_TransformerHead`) includes a real `nn.Dropout`
layer specifically so `08`'s MC Dropout has something to sample over —
this was a hard requirement on this architecture, not an optional extra.
`self.backbone` / `self.transformer_head` are named deliberately so
`get_param_groups(model, head_attr="transformer_head")` treats
*everything* freshly-initialized (projection, CLS token, positional
embeddings, encoder, classifier) as "the head" for discriminative-LR
purposes — not just the final linear layer, since none of those
components carry pretrained weights the way the ResNet backbone does.

## Results
Pending a real training run against the actual HAM10000 dataset — this
ADR documents and justifies the recipe/architecture decisions and the
code implementing them (unit- and integration-tested against synthetic
data: shape checks, freezing/param-group partition correctness,
early-stopping logic, and a full `fit()` smoke test all pass — see PR).
It does not claim empirical numbers that don't exist yet. Once `06` and
`07` are re-run: update this section with actual val_loss / val_accuracy
/ malignant_recall for both models, and the direct comparison table
`07`'s last cells produce automatically from `models/baseline_metrics.json`
and `models/champion_metrics.json`.

## Consequences
**Positive:**
- The baseline-vs-champion comparison in `07` now isolates the
  architecture as the variable, not a confound between "champion got a
  better recipe" and "champion is a better architecture."
- Early stopping means a run that isn't improving no longer silently
  burns compute for its full epoch budget.
- `ChampionModel` satisfies `08`'s MC Dropout requirement by
  construction, instead of that gap surfacing only when `08` is written.

**Trade-offs:**
- Two-phase training with discriminative LR groups is more moving parts
  than a single optimizer call — more to explain in review, more
  surface area for a future bug. Justified here because the overfitting
  it fixes is directly visible in `06`'s original training curve, not a
  theoretical concern.
- `WARMUP_EPOCHS`, `HEAD_LR`, `BACKBONE_LR`, `WEIGHT_DECAY`, and the
  scheduler/early-stopping settings are reasonable starting points, not
  exhaustively tuned (same caveat ADR-005 makes about its own
  hyperparameters). Revisit if `07`'s real results don't clearly improve
  on `06`'s.
- `d_model=256` / 3 encoder layers / 8 heads for the champion model were
  chosen as a moderate, trainable-on-limited-compute starting point, not
  swept. If compute allows (see Step 2's GPU/mixed-precision
  recommendation), worth trying larger before concluding the
  architecture itself doesn't help.
