# ADR-004: Malignant/Benign Risk Grouping

## Status
Accepted

## Context
The project's safety metric — recall on the malignant class — requires a
fixed definition of which of the 7 HAM10000 diagnoses count as "malignant."
This isn't just an EDA convenience; every notebook from `06_baseline_model`
onward, plus the final 3-tier threshold system and the cost-sensitive
business case, depends on this exact grouping being applied identically
everywhere.

One class is genuinely ambiguous: `akiec` (actinic keratosis) is
precancerous, not confirmed malignant. Different papers group it
differently — some treat it as its own "borderline" category.

## Decision
```
malignant = {mel, bcc, akiec}
benign    = {nv, bkl, df, vasc}
```

`akiec` is placed in the malignant bucket despite being precancerous
rather than confirmed cancer. This is a deliberate safety-conservative
choice: this system's whole purpose is flagging things that need a
dermatologist's attention. Treating a precancerous lesion as "benign"
risks a missed referral — the exact failure mode the project exists to
prevent. A false alarm on `akiec` costs a follow-up appointment; a missed
one costs delayed treatment.

Defined once in `src/labels.py` (`ALL_CLASSES`, `MALIGNANT_CLASSES`,
`BENIGN_CLASSES`, `risk_group()`). Every notebook and the eventual FastAPI
backend import this — none redefine it locally.

## Results
Computed on the training split in `03_eda.ipynb`: 19.4% malignant (1,356
images), 80.6% benign (5,625 images).

## Consequences
**Positive:**
- Single source of truth — the malignant/benign line can't silently
  drift between notebooks or between training and serving code.
- The `akiec` judgment call is documented and defensible, not an
  unstated assumption buried in one file.

**Trade-offs:**
- This is a project design choice, not a settled medical fact. Other
  reasonable groupings exist (e.g., treating `akiec` as its own
  intermediate category). If the report or viva calls for adjusting it,
  changing `src/labels.py` in one place propagates everywhere correctly
  — but the historical results in earlier notebooks were computed under
  this definition and would need re-running.
