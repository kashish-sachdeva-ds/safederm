# ADR-005: Class Imbalance Handling (Weighted Loss)

## Status
Accepted

## Context
Confirmed in `03_eda.ipynb`: the training split is dominated by `nv`
(67.1%), with rare classes as low as 1.0% (`df`). A model trained on
plain, unweighted cross-entropy loss can reach deceptively high accuracy
by defaulting toward the majority class while performing poorly on rare
and malignant classes — precisely the classes the project's actual
safety metric (recall on malignant, ADR-004) depends on.

## Decision
Applied inverse-frequency class weighting to the loss function:

```
weight[c] = total_samples / (num_classes * count[c])
```

Computed once in `src/engine.py::compute_class_weights()`, applied via
`nn.CrossEntropyLoss(weight=class_weights)` in `06_baseline_model.ipynb`.

A direct consequence: checkpoint selection during training uses weighted
validation loss as the "best" criterion, not raw accuracy. See Results
for why that distinction mattered concretely, not just in theory.

## Results
Computed weights on the training split (order matches `ALL_CLASSES`):

| Class | Weight |
|---|---|
| akiec | 4.492 |
| bcc | 2.763 |
| bkl | 1.292 |
| df | 14.046 |
| mel | 1.290 |
| nv | 0.213 |
| vasc | 10.074 |

During training, epoch 8 reached higher raw accuracy (79.4%) than epoch
5 (77.4%), but epoch 5 had lower weighted val loss and was correctly
selected as "best." On the selected checkpoint: malignant recall reached
83.1% (255/308 correctly flagged), while `nv` precision stayed high
(0.96) despite `nv`'s weight being deliberately suppressed to 0.213.

## Consequences
**Positive:**
- Malignant classes (`mel`, `bcc`, `akiec`) individually reached 0.72–0.83
  recall — far above what an unweighted baseline would likely achieve
  given how small a share of the data they are.
- Checkpoint selection is aligned with the project's actual priority
  (balanced/malignant sensitivity), not raw accuracy, which can hide
  poor minority-class performance behind a dominant `nv` class.

**Trade-offs:**
- `nv` recall dropped to 0.80 — a deliberate trade, sacrificing some
  majority-class precision-on-recall for minority-class sensitivity.
- 52 malignant cases (17%) were still missed in this baseline. Weighting
  the loss reduces, but does not eliminate, missed malignant cases —
  `07_champion_model.ipynb` (better architecture) and
  `08_calibration_conformal.ipynb` (threshold tuning) are expected to
  improve this further, not fully solve it here.
- Alternative techniques (focal loss, oversampling/SMOTE, undersampling
  the majority class) were not tried in this baseline. Inverse-frequency
  weighting was chosen as the simplest, most standard first approach —
  worth revisiting if `07`'s results don't improve enough on malignant
  recall specifically.
