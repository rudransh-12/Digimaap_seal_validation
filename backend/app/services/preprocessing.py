"""
SealScan -- Image Preprocessing Service.

Produces a consistent PreprocessedImage bundle used by feature extraction.
The same pipeline is applied to both the current image and every reference image.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.config.settings import PREPROCESSING
from app.utils.image_utils import to_grayscale, to_hsv

logger = logging.getLogger("sealscan.preprocessing")


@dataclass
class PreprocessedImage:
    """All colour/greyscale representations produced during preprocessing."""
    bgr: np.ndarray           # resized BGR (for colour ops)
    gray: np.ndarray          # greyscale (for ORB, SSIM, sharpness)
    hsv: np.ndarray           # HSV (for histogram comparison)
    edges: np.ndarray         # Canny edge map (uint8, 0/255)
    gray_norm: np.ndarray     # normalised float greyscale in [0,1] (for cosine)


class ImagePreprocessor:
    """Stateless image preprocessing pipeline."""

    def __init__(self) -> None:
        cfg = PREPROCESSING
        self._size: tuple[int, int] = tuple(cfg["target_size"])   # (w, h)
        self._blur_k: int = cfg["gaussian_blur_kernel"]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def process(self, bgr_raw: np.ndarray) -> PreprocessedImage:
        """
        Run the full preprocessing pipeline on a raw BGR image.

        Steps:
          1. Resize to standard working resolution.
          2. Optional Gaussian denoising.
          3. Convert to greyscale.
          4. Convert to HSV for histogram features.
          5. Normalise greyscale to [0, 1] float.
          6. Generate Canny edge map.
        """
        # 1. Resize
        bgr = cv2.resize(bgr_raw, self._size, interpolation=cv2.INTER_AREA)

        # 2. Denoise
        if self._blur_k and self._blur_k > 1:
            k = self._blur_k if self._blur_k % 2 == 1 else self._blur_k + 1
            bgr = cv2.GaussianBlur(bgr, (k, k), 0)

        # 3. Greyscale
        gray = to_grayscale(bgr)

        # 4. HSV
        hsv = to_hsv(bgr)

        # 5. Normalised greyscale
        gray_norm = gray.astype(np.float32) / 255.0

        # 6. Edge map -- auto-threshold via Otsu.
        # cv2.threshold returns (threshold_value, thresholded_image).
        # First element is the float threshold; second is the binary mask image.
        otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t_low = float(otsu_val) * 0.5 if float(otsu_val) > 0 else 50.0
        t_high = float(otsu_val) if float(otsu_val) > 0 else 150.0
        edges = cv2.Canny(gray, t_low, t_high)

        logger.debug(
            "Preprocessed | size=%s gray_mean=%.2f edge_density=%.4f",
            self._size,
            float(np.mean(gray)),
            float(np.mean(edges > 0)),
        )

        return PreprocessedImage(
            bgr=bgr,
            gray=gray,
            hsv=hsv,
            edges=edges,
            gray_norm=gray_norm,
        )
