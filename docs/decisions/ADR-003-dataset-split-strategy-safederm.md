# ADR-003: Dataset Split Strategy

## Status
Accepted

## Context
HAM10000 contains 10,015 images but only 7,470 unique lesions — 1,956 lesions
have more than one photo (different angle, zoom, or follow-up visit of the
same mole). Confirmed directly against the metadata in `01_data_extraction.ipynb`.

If the same lesion's photos can land in both train and test, the model can
partly memorize that specific lesion's texture and lighting instead of
learning to recognize the diagnostic class in general. Test performance would
look better than the model actually deserves — exactly the kind of inflated
result this project is trying not to produce.

Note: HAM10000's metadata provides `lesion_id`, not a true patient ID. Some
patients may have multiple distinct lesions in the dataset, and those aren't
linked to each other. So this is a **lesion-level split**, not a strict
patient-level split — it closes the specific leakage path we can detect and
control for, but it isn't a guarantee against all possible patient-level
overlap.

## Decision
Split by `lesion_id`, not `image_id`:

1. Group metadata by `lesion_id`, take each lesion's diagnosis (`dx`) once.
2. Split lesions 70% train / 15% val / 15% test, stratified by `dx` so rare
   classes (`df`: 73 lesions, `vasc`: 98 lesions) stay represented in every
   split.
3. `random_state=42` — split is reproducible; every teammate gets the same
   assignment.
4. Map the lesion-level split back to image-level rows. Every photo of a
   given lesion follows that lesion into exactly one split.
5. Hard assertion: zero `lesion_id` overlap between any two splits, checked
   before saving. The notebook fails loudly if this is ever violated.

Implemented in `notebooks/02_data_understanding.ipynb`.

## Results
| Split | Lesions | Images | nv % | mel % | df % |
|---|---|---|---|---|---|
| Train | 5,229 | 6,981 | 67.1 | 11.1 | 1.0 |
| Val | 1,120 | 1,532 | 66.4 | 11.3 | 1.6 |
| Test | 1,121 | 1,502 | 66.8 | 11.1 | 1.3 |

No lesion appears in more than one split. Class proportions stay consistent
across all three (within ~1-2 points), including the rarest class.

## Consequences
**Positive:**
- No lesion-level leakage between train/val/test — the model's test score
  reflects generalization, not memorization.
- Reproducible: same split every run, same split for every teammate.
- Rare classes remain present in val and test, so calibration and final
  metrics aren't computed on zero or near-zero samples of any class.

**Trade-offs / accepted limitations:**
- Image counts per split aren't perfectly proportional to the 70/15/15
  target (val: 1,532 images, test: 1,502) because lesion group sizes vary.
  This is expected and not adjusted for — forcing exact image-count balance
  would require splitting by image again, reintroducing the leakage this
  ADR exists to prevent.
- This guards against lesion-level leakage specifically, not all
  patient-level leakage, since the dataset doesn't expose a linking patient
  ID beyond `lesion_id`. Documented here so it isn't overstated later in the
  report or viva.
- 57 rows have missing `age`. Not handled in this split — deferred to the
  feature engineering stage, since `age` isn't used for splitting or
  stratification here.
