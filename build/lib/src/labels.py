"""Diagnostic class definitions and risk grouping for SafeDerm.

Single source of truth for the malignant/benign grouping used throughout
the project -- the model's safety metric (recall on malignant) depends on
this exact grouping being applied consistently everywhere it's used.
See ADR-004 for the rationale.
"""

# All 7 HAM10000 diagnostic classes, fixed order -- this order also defines
# the integer label encoding used by src/dataset.py.
ALL_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Malignant: requires clinical follow-up/referral.
# akiec (actinic keratosis) is precancerous, not confirmed cancer -- included
# here as the safety-conservative choice. See ADR-004.
MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}

BENIGN_CLASSES = set(ALL_CLASSES) - MALIGNANT_CLASSES


def risk_group(dx: str) -> str:
    """Maps a diagnosis code to 'malignant' or 'benign'."""
    return "malignant" if dx in MALIGNANT_CLASSES else "benign"
