@echo off
set SAFEDERM_MODEL_VARIANT=baseline
echo Starting SafeDerm API with BASELINE model...
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
