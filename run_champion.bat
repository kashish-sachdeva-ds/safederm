@echo off
set SAFEDERM_MODEL_VARIANT=champion
echo Starting SafeDerm API with CHAMPION model...
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
