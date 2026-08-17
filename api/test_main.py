import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


from unittest.mock import patch

@pytest.fixture(scope="module")
def client():
    """One TestClient (and one model-loading lifespan run) shared across
    every test in this module, instead of each test function spinning up
    its own -- previously every test re-triggered startup, which now
    means re-constructing a full ResNet-50 each time.
    """
    with patch("api.main.verify_image_is_skin", return_value=(True, 0.99)), \
         patch("api.main.verify_in_distribution", return_value=(True, 0.5)):
        with TestClient(app) as c:
            yield c


def _dummy_image_bytes() -> bytes:
    image = Image.new("RGB", (224, 224), color=(200, 150, 100))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    # checkpoint_loaded / calibration_loaded are the honesty fix for the
    # old model_loaded-conflates-two-things bug: model_loaded is True the
    # instant the architecture constructs, even with random weights, so a
    # regression back to a single "model_loaded" field would silently
    # reintroduce that. Assert the fields exist with the right type,
    # rather than a fixed value -- whether a checkpoint is actually
    # present depends on where this suite runs (CI vs. someone's machine
    # post-training).
    assert isinstance(data["checkpoint_loaded"], bool)
    assert isinstance(data["calibration_loaded"], bool)
    assert isinstance(data["feature_bank_loaded"], bool)
    assert data["model_variant"] in ("baseline", "champion")


def test_predict_invalid_file_type(client):
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"This is a text file, not an image.", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "File provided is not an image."


def test_predict_file_too_large(client):
    large_content = b"0" * (6 * 1024 * 1024)
    response = client.post(
        "/predict",
        files={"file": ("large_image.jpg", large_content, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "too large" in response.json()["detail"].lower()


def test_predict_corrupt_image_does_not_leak_internals(client):
    """A malformed file with an image content-type should 500 with a
    generic message -- not the raw PIL/library exception text. Directly
    tests the fix for the old `detail=f"Error processing image: {e}"`
    pattern, which returned internal exception details straight to the
    client.
    """
    corrupt = b"\xff\xd8\xff\xe0this is not a valid jpeg body"
    response = client.post(
        "/predict",
        files={"file": ("corrupt.jpg", corrupt, "image/jpeg")},
    )
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "Truncated" not in detail  # the actual PIL error text must not appear
    assert "traceback" not in detail.lower()
    assert detail == "Internal error while processing the image. Please try again."


def test_predict_valid_dummy_image(client):
    response = client.post(
        "/predict",
        files={"file": ("dummy.jpg", _dummy_image_bytes(), "image/jpeg")},
    )
    # No more "or 500 is also fine" escape hatch: a well-formed image
    # should always succeed. A test that passes whether the endpoint
    # works or crashes isn't testing the endpoint.
    assert response.status_code == 200
    data = response.json()
    assert data["diagnosis"] in {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
    assert data["risk_group"] in {"malignant", "benign"}
    assert data["tier"] in {"normal", "concerning", "uncertain", "not_calibrated"}
    assert isinstance(data["calibrated"], bool)
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 7
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-3
    assert data["model_variant"] in ("baseline", "champion")


def test_cors_rejects_non_allowlisted_origin(client):
    """The old `allow_origins=["*"], allow_credentials=True` combination
    let Starlette reflect any request's Origin header back, effectively
    allowing credentialed requests from anywhere. This checks a
    not-allow-listed origin is NOT reflected in the response.
    """
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://not-an-allowed-origin.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in {k.lower() for k in response.headers.keys()} \
        or response.headers.get("access-control-allow-origin") != "http://not-an-allowed-origin.example.com"


def test_predict_succeeds_without_feature_bank(client):
    """No models/feature_bank_<variant>.pt exists in this test run (none is
    committed to the repo) -- Gateway 2 must be skipped gracefully, not
    crash the request or silently report success it didn't perform. This
    is the live regression test for the old load_feature_bank() crash:
    if that bug came back, this whole module would fail to collect before
    this test could even run.
    """
    health = client.get("/health").json()
    
    from pathlib import Path
    has_feature_bank = Path("models/feature_bank.pt").exists() or Path("models/feature_bank_baseline.pt").exists()
    if not has_feature_bank:
        assert health["feature_bank_loaded"] is False

    response = client.post(
        "/predict",
        files={"file": ("dummy.jpg", _dummy_image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
