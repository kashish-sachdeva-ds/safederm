# SafeDerm

A calibrated CNN-Transformer hybrid for uncertainty-aware skin lesion triage in primary care clinics.

## Problem
Primary care clinics lack dermatology expertise. Patients with benign lesions are unnecessarily referred to specialists, overloading hospitals like PGI and AIIMS. Dangerous cases can also be missed.

## Solution
SafeDerm triages skin lesion images into three categories:
- **Routine monitoring** (high confidence benign)
- **Urgent referral** (high confidence malignant)
- **Uncertain — needs human review** (low confidence, safety-first)


## Setup
```bash
git clone https://github.com/yourname/safederm.git
cd safederm
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt