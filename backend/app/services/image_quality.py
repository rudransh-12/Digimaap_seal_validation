"""
SealScan -- Image Quality Service.

Evaluates a raw BGR image against configurable thresholds and returns
a structured result used by BOTH endpoints.
"""
from __future__ import annotations
import logging

import cv2
import numpy as np

from app.config.settings import QUALITY_THRESHOLDS
from app.models.schemas import MetricDetail, AllMetrics, ResolutionDetail
from app.utils.image_utils import to_grayscale

logger = logging.getLogger("sealscan.image_quality")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _measure_sharpness(gray: np.ndarray) -> float:
    """Laplacian variance -- higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _measure_brightness(gray: np.ndarray) -> float:
    """Mean pixel intensity (0-255)."""
    return float(np.mean(gray))


def _measure_contrast(gray: np.ndarray) -> float:
    """Standard deviation of pixel intensities."""
    return float(np.std(gray))


def _estimate_noise(gray: np.ndarray) -> float:
    """
    Estimate noise using the high-frequency residual after Gaussian blur.
    Returns the standard deviation of the difference image.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    diff = gray.astype(np.float32) - blurred.astype(np.float32)
    return float(np.std(diff))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_quality(bgr: np.ndarray) -> tuple[bool, AllMetrics, list[MetricDetail]]:
    """
    Run all quality checks against QUALITY_THRESHOLDS.

    Returns:
        passed  (bool)          -- True if ALL checks pass
        metrics (AllMetrics)    -- full detail for every metric (success response)
        failed  (list[MetricDetail]) -- only the failing metrics (failure response)
    """
    t = QUALITY_THRESHOLDS
    gray = to_grayscale(bgr)

    h, w = bgr.shape[:2]
    sharpness = _measure_sharpness(gray)
    brightness = _measure_brightness(gray)
    contrast = _measure_contrast(gray)
    noise = _estimate_noise(gray)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    res_passed = (w >= t["min_width"]) and (h >= t["min_height"])
    res_detail = ResolutionDetail(width=w, height=h, passed=res_passed)

    # ------------------------------------------------------------------
    # Sharpness
    # ------------------------------------------------------------------
    sharp_passed = sharpness >= t["min_sharpness"]
    sharp_detail = MetricDetail(
        metric="sharpness",
        value=round(sharpness, 4),
        threshold=t["min_sharpness"],
        passed=sharp_passed,
        message=None if sharp_passed else "Image is too blurry. Please retake with a steadier hand.",
    )

    # ------------------------------------------------------------------
    # Brightness
    # ------------------------------------------------------------------
    bright_passed = t["min_brightness"] <= brightness <= t["max_brightness"]
    if brightness < t["min_brightness"]:
        bright_msg = "Image is too dark. Improve lighting conditions."
    elif brightness > t["max_brightness"]:
        bright_msg = "Image is overexposed. Reduce light source or adjust camera."
    else:
        bright_msg = None
    bright_detail = MetricDetail(
        metric="brightness",
        value=round(brightness, 4),
        threshold=t["max_brightness"] if brightness > t["max_brightness"] else t["min_brightness"],
        passed=bright_passed,
        message=bright_msg,
    )

    # ------------------------------------------------------------------
    # Contrast
    # ------------------------------------------------------------------
    contrast_passed = contrast >= t["min_contrast"]
    contrast_detail = MetricDetail(
        metric="contrast",
        value=round(contrast, 4),
        threshold=t["min_contrast"],
        passed=contrast_passed,
        message=None if contrast_passed else "Image contrast is too low. Ensure the seal is well lit.",
    )

    # ------------------------------------------------------------------
    # Noise
    # ------------------------------------------------------------------
    noise_passed = noise <= t["max_noise"]
    noise_detail = MetricDetail(
        metric="noise",
        value=round(noise, 4),
        threshold=t["max_noise"],
        passed=noise_passed,
        message=None if noise_passed else "Excessive image noise detected. Use better lighting or a higher quality camera.",
    )

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    all_metrics = AllMetrics(
        resolution=res_detail,
        sharpness=sharp_detail,
        brightness=bright_detail,
        contrast=contrast_detail,
        noise=noise_detail,
    )

    failed: list[MetricDetail] = []
    if not res_passed:
        failed.append(
            MetricDetail(
                metric="resolution",
                value=float(min(w, h)),
                threshold=float(min(t["min_width"], t["min_height"])),
                passed=False,
                message=f"Image resolution {w}x{h} is below the minimum {t['min_width']}x{t['min_height']}.",
            )
        )
    for d in [sharp_detail, bright_detail, contrast_detail, noise_detail]:
        if not d.passed:
            failed.append(d)

    passed = len(failed) == 0

    logger.info(
        "Quality check | passed=%s | sharpness=%.2f brightness=%.2f contrast=%.2f noise=%.2f res=%dx%d",
        passed, sharpness, brightness, contrast, noise, w, h,
    )

    return passed, all_metrics, failed
