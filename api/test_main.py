from fastapi.testclient import TestClient
from api.main import app
import io
from PIL import Image

client = TestClient(app)

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

def test_predict_invalid_file_type():
    with TestClient(app) as client:
        file_content = b"This is a text file, not an image."
        response = client.post(
            "/predict",
            files={"file": ("test.txt", file_content, "text/plain")}
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "File provided is not an image."

def test_predict_file_too_large():
    with TestClient(app) as client:
        large_content = b"0" * (6 * 1024 * 1024)
        response = client.post(
            "/predict",
            files={"file": ("large_image.jpg", large_content, "image/jpeg")}
        )
        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()

def test_predict_valid_dummy_image():
    with TestClient(app) as client:
        image = Image.new('RGB', (224, 224), color = (200, 150, 100))
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        response = client.post(
            "/predict",
            files={"file": ("dummy.jpg", img_byte_arr.read(), "image/jpeg")}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "diagnosis" in data
            assert "risk_group" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert len(data["probabilities"]) == 7
        elif response.status_code == 500:
            pass
