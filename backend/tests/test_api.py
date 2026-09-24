"""
Integration tests for the FastAPI endpoints using TestClient.
"""
import io
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _png_bytes(width: int = 512, height: int = 512, variant: int = 0) -> bytes:
    """
    Generate a realistic-looking synthetic grayscale PNG that passes ALL quality checks.
    Uses a horizontal gradient + horizontal stripe texture + mild noise.
    'variant' shifts the texture phase so different calls produce distinct images.
    """
    rng = np.random.default_rng(42 + variant)
    x = np.linspace(0, 255, width).astype(np.uint8)
    base = np.tile(x, (height, 1))
    for i in range(variant * 7, height, 32):
        base[i:i + 16, :] = np.clip(base[i:i + 16, :].astype(np.int32) + 40, 0, 255).astype(np.uint8)
    noise = rng.integers(-5, 6, (height, width)).astype(np.int16)
    gray = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    bgr = np.stack([gray, gray, gray], axis=-1)
    _, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


def _blurry_png_bytes(width: int = 512, height: int = 512) -> bytes:
    """Constant-value image -> zero sharpness."""

    bgr = np.full((height, width, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# POST /quality-check
# ---------------------------------------------------------------------------
class TestQualityCheckEndpoint:
    def test_quality_check_pass(self):
        data = _png_bytes()
        response = client.post(
            "/quality-check",
            files={"image": ("test.png", io.BytesIO(data), "image/png")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["quality_passed"] is True
        assert "metrics" in body

    def test_quality_check_blurry_fails(self):
        data = _blurry_png_bytes()
        response = client.post(
            "/quality-check",
            files={"image": ("blurry.png", io.BytesIO(data), "image/png")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["quality_passed"] is False
        assert len(body["failed_metrics"]) >= 1

    def test_quality_check_missing_file(self):
        response = client.post("/quality-check")
        assert response.status_code == 422

    def test_quality_check_invalid_extension(self):
        response = client.post(
            "/quality-check",
            files={"image": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert response.status_code == 422

    def test_quality_check_empty_file(self):
        response = client.post(
            "/quality-check",
            files={"image": ("empty.png", io.BytesIO(b""), "image/png")},
        )
        assert response.status_code == 422

    def test_quality_check_metrics_structure(self):
        data = _png_bytes()
        response = client.post(
            "/quality-check",
            files={"image": ("test.png", io.BytesIO(data), "image/png")},
        )
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
        cur = _png_bytes()
        ref = _png_bytes()
        response = client.post(
            "/seal-scan/similarity",
            files=[
                ("current_image", ("current.png", io.BytesIO(cur), "image/png")),
                ("reference_images", ("ref1.png", io.BytesIO(ref), "image/png")),
            ],
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["quality_passed"] is True
        assert "tampering_assessment" in body
        assert "aggregated_metrics" in body
        assert "reference_comparisons" in body

    def test_similarity_multiple_references(self):
        cur = _png_bytes()
        ref1 = _png_bytes()
        ref2 = _png_bytes(variant=1)
        response = client.post(
            "/seal-scan/similarity",
            files=[
                ("current_image", ("current.png", io.BytesIO(cur), "image/png")),
                ("reference_images", ("ref1.png", io.BytesIO(ref1), "image/png")),
                ("reference_images", ("ref2.png", io.BytesIO(ref2), "image/png")),
            ],
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["reference_comparisons"]) == 2
        assert body["reference_comparisons"][0]["reference_id"] == "reference_1"
        assert body["reference_comparisons"][1]["reference_id"] == "reference_2"

    def test_similarity_quality_fail_returns_failed_metrics(self):
        cur = _blurry_png_bytes()
        ref = _png_bytes()
        response = client.post(
            "/seal-scan/similarity",
            files=[
                ("current_image", ("blurry.png", io.BytesIO(cur), "image/png")),
                ("reference_images", ("ref.png", io.BytesIO(ref), "image/png")),
            ],
        )
        assert response.status_code == 200
        body = response.json()
        assert body["quality_passed"] is False
        assert body["success"] is False
        assert "failed_metrics" in body
        # No similarity data when quality fails
        assert "tampering_assessment" not in body

    def test_similarity_missing_reference(self):
        cur = _png_bytes()
        response = client.post(
            "/seal-scan/similarity",
            files=[
                ("current_image", ("current.png", io.BytesIO(cur), "image/png")),
            ],
        )
        # Should get 422 for missing reference_images
        assert response.status_code == 422

    def test_tampering_assessment_structure(self):
        cur = _png_bytes()
        ref = _png_bytes()
        response = client.post(
            "/seal-scan/similarity",
            files=[
                ("current_image", ("current.png", io.BytesIO(cur), "image/png")),
                ("reference_images", ("ref.png", io.BytesIO(ref), "image/png")),
            ],
        )
        body = response.json()
        if body.get("success"):
            ta = body["tampering_assessment"]
            assert "tampering_score" in ta
            assert "risk_class" in ta
            assert ta["risk_class"] in ["LOW", "MEDIUM", "HIGH"]
            assert 0.0 <= ta["tampering_score"] <= 1.0

    def test_aggregated_metrics_keys(self):
        cur = _png_bytes()
        ref = _png_bytes()
        response = client.post(
            "/seal-scan/similarity",
            files=[
                ("current_image", ("current.png", io.BytesIO(cur), "image/png")),
                ("reference_images", ("ref.png", io.BytesIO(ref), "image/png")),
            ],
        )
        body = response.json()
        if body.get("success"):
            am = body["aggregated_metrics"]
            for key in ["cosine_similarity", "orb_match_ratio", "ssim_score",
                        "edge_difference", "histogram_difference", "shape_difference"]:
                assert key in am


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
