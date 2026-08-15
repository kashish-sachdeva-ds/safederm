FROM python:3.13-slim

WORKDIR /app

# Copy requirements files first to leverage Docker layer caching -- this
# layer only invalidates when dependencies change, not on every code edit.
COPY requirements.txt .
COPY api/requirements.txt ./api/

# Two separate installs on purpose: the first (root deps: torch, pandas,
# etc.) rarely changes and stays cached; the second (api/requirements.txt,
# now self-sufficient via `-r ../requirements.txt` -- see that file) only
# reinstalls fastapi/uvicorn/python-multipart on top, since pip sees the
# root deps are already satisfied. requirements.txt no longer pins a
# +cuXXX torch build, so this installs a CPU wheel -- correct for this
# image, since docker-compose.yml requests no GPU device.
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy the rest of the application
COPY src/ ./src/
COPY api/ ./api/

EXPOSE 8000

# Hits the app's own /health endpoint -- fails the container health check
# if the process is up but stuck (e.g. startup crashed before serving).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

ENTRYPOINT ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
