FROM python:3.13-slim

WORKDIR /app

# Install system dependencies required for OpenCV or other ML libraries if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files first to leverage Docker cache
COPY requirements.txt .
COPY api/requirements.txt ./api/

# Install dependencies (ignoring the large ML dependencies if they are not needed, but we do need them here)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy the rest of the application
COPY src/ ./src/
COPY api/ ./api/

# Expose port for FastAPI
EXPOSE 8000

# Run the API
ENTRYPOINT ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
