import requests
from pathlib import Path

# Pick a real image from the training set
img_path = next(Path("data/raw/images").glob("*.jpg"))
print(f"Testing real image: {img_path}")

url = "http://127.0.0.1:8000/predict"
with open(img_path, "rb") as f:
    files = {"file": (img_path.name, f, "image/jpeg")}
    response = requests.post(url, files=files)

print(response.status_code)
print(response.json())
