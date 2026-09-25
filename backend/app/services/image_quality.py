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


def _measure_brenner_sharpness(gray: np.ndarray) -> float:
    """
    Brenner gradient focus measure -- higher = sharper.
    Computes mean squared difference between pixels separated by 2 units along both axes.
    """
    h, w = gray.shape
    if w <= 2 or h <= 2:
        return 0.0

    diff_x = (gray[:, 2:].astype(np.float64) - gray[:, :-2].astype(np.float64)) ** 2
    diff_y = (gray[2:, :].astype(np.float64) - gray[:-2, :].astype(np.float64)) ** 2
    return float(0.5 * (np.mean(diff_x) + np.mean(diff_y)))


def _measure_canny_edge_density(gray: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> float:
    """
    Ratio of edge pixels detected by Canny detector to total image pixels.
    Returns a float in [0.0, 1.0].
    """
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return float(np.count_nonzero(edges) / edges.size)


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
    brenner = _measure_brenner_sharpness(gray)
    canny_density = _measure_canny_edge_density(gray)
    brightness = _measure_brightness(gray)
    contrast = _measure_contrast(gray)
    noise = _estimate_noise(gray)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    res_passed = (w >= t["min_width"]) and (h >= t["min_height"])
    res_detail = ResolutionDetail(width=w, height=h, passed=res_passed)

    # ------------------------------------------------------------------
    # Sharpness (Laplacian)
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
    # Brenner Sharpness
    # ------------------------------------------------------------------
    brenner_passed = brenner >= t["min_brenner_sharpness"]
    brenner_detail = MetricDetail(
        metric="brenner_sharpness",
        value=round(brenner, 4),
        threshold=t["min_brenner_sharpness"],
        passed=brenner_passed,
        message=None if brenner_passed else "Image focus is insufficient according to Brenner gradient.",
    )

    # ------------------------------------------------------------------
    # Canny Edge Density
    # ------------------------------------------------------------------
    canny_passed = canny_density >= t["min_canny_edge_density"]
    canny_detail = MetricDetail(
        metric="canny_edge_density",
        value=round(canny_density, 4),
        threshold=t["min_canny_edge_density"],
        passed=canny_passed,
        message=None if canny_passed else "Insufficient edge details detected in the seal image.",
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
        brenner_sharpness=brenner_detail,
        canny_edge_density=canny_detail,
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
    for d in [sharp_detail, brenner_detail, canny_detail, bright_detail, contrast_detail, noise_detail]:
        if not d.passed:
            failed.append(d)

    passed = len(failed) == 0

    logger.info(
        "Quality check | passed=%s | laplacian=%.2f brenner=%.2f canny=%.4f brightness=%.2f contrast=%.2f noise=%.2f res=%dx%d",
        passed, sharpness, brenner, canny_density, brightness, contrast, noise, w, h,
    )

    return passed, all_metrics, failed
