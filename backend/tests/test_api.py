"""
Integration tests for the FastAPI endpoints using TestClient.
Now uses JSON body with base64-encoded images.
"""
import base64
import io
import json
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _b64(variant: int = 0) -> str:
    """
    Generate a realistic synthetic grayscale image and return it as a
    plain base64 string (no data-URI prefix).
    """
    rng = np.random.default_rng(42 + variant)
    x = np.linspace(0, 255, 512).astype(np.uint8)
    base = np.tile(x, (512, 1))
    for i in range(variant * 7, 512, 32):
        base[i:i + 16, :] = np.clip(base[i:i + 16, :].astype(np.int32) + 40, 0, 255).astype(np.uint8)
    noise = rng.integers(-5, 6, (512, 512)).astype(np.int16)
    gray = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    bgr = np.stack([gray, gray, gray], axis=-1)
    _, buf = cv2.imencode(".jpg", bgr)
    return base64.b64encode(buf.tobytes()).decode()


def _blurry_b64() -> str:
    """Constant-value image -> zero sharpness."""
    bgr = np.full((512, 512, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", bgr)
    return base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------------------
# POST /quality-check
# ---------------------------------------------------------------------------
class TestQualityCheckEndpoint:
    def test_quality_check_pass(self):
        response = client.post(
            "/quality-check",
            json={"image": _b64()},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["quality_passed"] is True
        assert "metrics" in body

    def test_quality_check_blurry_fails(self):
        response = client.post(
            "/quality-check",
            json={"image": _blurry_b64()},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["quality_passed"] is False
        assert len(body["failed_metrics"]) >= 1

    def test_quality_check_missing_field(self):
        response = client.post("/quality-check", json={})
        assert response.status_code == 422

    def test_quality_check_empty_string(self):
        response = client.post("/quality-check", json={"image": ""})
        assert response.status_code == 422

    def test_quality_check_invalid_base64(self):
        response = client.post("/quality-check", json={"image": "not_valid_base64!!!"})
        assert response.status_code == 422

    def test_quality_check_data_uri_prefix(self):
        """Flutter may send data:image/jpeg;base64,... — should be accepted."""
        b64 = _b64()
        response = client.post(
            "/quality-check",
            json={"image": f"data:image/jpeg;base64,{b64}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["quality_passed"] is True

    def test_quality_check_metrics_structure(self):
        response = client.post("/quality-check", json={"image": _b64()})
        body = response.json()
        if body["quality_passed"]:
            metrics = body["metrics"]
            assert "resolution" in metrics
            assert "sharpness" in metrics
            assert "brightness" in metrics
            assert "contrast" in metrics


# ---------------------------------------------------------------------------
# POST /seal-scan/similarity
# ---------------------------------------------------------------------------
class TestSimilarityEndpoint:
    def test_similarity_success(self):
        response = client.post(
            "/seal-scan/similarity",
            json={"current_image": _b64(), "reference_images": [_b64()]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["quality_passed"] is True
        assert "tampering_assessment" in body
        assert "aggregated_metrics" in body
        assert "reference_comparisons" in body

    def test_similarity_multiple_references(self):
        response = client.post(
            "/seal-scan/similarity",
            json={
                "current_image": _b64(),
                "reference_images": [_b64(), _b64(variant=1)],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["reference_comparisons"]) == 2
        assert body["reference_comparisons"][0]["reference_id"] == "reference_1"
        assert body["reference_comparisons"][1]["reference_id"] == "reference_2"

    def test_similarity_quality_fail_returns_failed_metrics(self):
        response = client.post(
            "/seal-scan/similarity",
            json={"current_image": _blurry_b64(), "reference_images": [_b64()]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["quality_passed"] is False
        assert body["success"] is False
        assert "failed_metrics" in body
        assert "tampering_assessment" not in body

    def test_similarity_missing_reference_images_field(self):
        response = client.post(
            "/seal-scan/similarity",
            json={"current_image": _b64()},
        )
        assert response.status_code == 422

    def test_similarity_empty_reference_list(self):
        """min_length=1 on reference_images field should reject empty list."""
        response = client.post(
            "/seal-scan/similarity",
            json={"current_image": _b64(), "reference_images": []},
        )
        assert response.status_code == 422

    def test_tampering_assessment_structure(self):
        response = client.post(
            "/seal-scan/similarity",
            json={"current_image": _b64(), "reference_images": [_b64()]},
        )
        body = response.json()
        if body.get("success"):
            ta = body["tampering_assessment"]
            assert "tampering_score" in ta
            assert "risk_class" in ta
            assert ta["risk_class"] in ["LOW", "MEDIUM", "HIGH"]
            assert 0.0 <= ta["tampering_score"] <= 1.0

    def test_aggregated_metrics_keys(self):
        response = client.post(
            "/seal-scan/similarity",
            json={"current_image": _b64(), "reference_images": [_b64()]},
        )
        body = response.json()
        if body.get("success"):
            am = body["aggregated_metrics"]
            for key in ["cosine_similarity", "orb_match_ratio", "ssim_score",
                        "edge_difference", "histogram_difference", "shape_difference"]:
                assert key in am

    def test_similarity_data_uri_current_image(self):
        """data-URI prefix on current_image should be accepted."""
        b64 = _b64()
        response = client.post(
            "/seal-scan/similarity",
            json={
                "current_image": f"data:image/jpeg;base64,{b64}",
                "reference_images": [_b64()],
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
