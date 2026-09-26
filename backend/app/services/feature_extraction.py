"""
SealScan -- Feature Extraction Service.

Computes the six similarity metrics between one pair of
(current, reference) PreprocessedImage objects.
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from app.config.settings import SIFT_CONFIG
from app.services.preprocessing import PreprocessedImage

logger = logging.getLogger("sealscan.feature_extraction")


@dataclass
class SimilarityMetrics:
    cosine_similarity: float
    sift_match_ratio: float
    ssim_score: float
    edge_difference: float
    histogram_difference: float
    shape_difference: float


class FeatureExtractor:
    """Computes all six similarity metrics between a pair of preprocessed images."""

    def __init__(self) -> None:
        cfg = SIFT_CONFIG
        self._sift = cv2.SIFT_create(
            nfeatures=cfg["n_features"],
            nOctaveLayers=cfg["n_octave_layers"],
            contrastThreshold=cfg["contrast_threshold"],
            edgeThreshold=cfg["edge_threshold"],
            sigma=cfg["sigma"],
        )
        # FLANN matcher for floating-point SIFT descriptors (KD-Tree index)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self._matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self._lowe = cfg["lowe_ratio"]
        self._ransac_thresh = cfg["ransac_reproj_threshold"]
        self._min_good = cfg["min_good_matches"]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def compute(self, current: PreprocessedImage, reference: PreprocessedImage) -> SimilarityMetrics:
        cos_sim = self._cosine_similarity(current.gray_norm, reference.gray_norm)
        sift_ratio = self._sift_match_ratio(current.gray, reference.gray)
        ssim_score = self._ssim_score(current.gray, reference.gray)
        edge_diff = self._edge_difference(current.edges, reference.edges)
        hist_diff = self._histogram_difference(current.hsv, reference.hsv)
        shape_diff = self._shape_difference(current.gray, reference.gray)

        metrics = SimilarityMetrics(
            cosine_similarity=round(cos_sim, 6),
            sift_match_ratio=round(sift_ratio, 6),
            ssim_score=round(ssim_score, 6),
            edge_difference=round(edge_diff, 6),
            histogram_difference=round(hist_diff, 6),
            shape_difference=round(shape_diff, 6),
        )
        logger.debug("Metrics | %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # A. Cosine Similarity
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        flat_a = a.flatten().astype(np.float64)
        flat_b = b.flatten().astype(np.float64)
        norm_a = np.linalg.norm(flat_a)
        norm_b = np.linalg.norm(flat_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(flat_a, flat_b) / (norm_a * norm_b))

    # ------------------------------------------------------------------
    # B. SIFT Match Ratio
    # ------------------------------------------------------------------
    def _sift_match_ratio(self, gray_cur: np.ndarray, gray_ref: np.ndarray) -> float:
        kp_cur, des_cur = self._sift.detectAndCompute(gray_cur, None)
        kp_ref, des_ref = self._sift.detectAndCompute(gray_ref, None)

        if (
            des_cur is None
            or des_ref is None
            or len(kp_cur) < 2
            or len(kp_ref) < 2
        ):
            return 0.0

        # Ensure float32 format for FLANN KD-Tree matcher
        if des_cur.dtype != np.float32:
            des_cur = des_cur.astype(np.float32)
        if des_ref.dtype != np.float32:
            des_ref = des_ref.astype(np.float32)

        # Lowe's ratio test (kNN k=2)
        try:
            matches = self._matcher.knnMatch(des_cur, des_ref, k=2)
        except cv2.error as err:
            logger.warning("FLANN matching error: %s. Returning 0.0", err)
            return 0.0

        good: list[cv2.DMatch] = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self._lowe * n.distance:
                    good.append(m)

        # Optional RANSAC geometric verification
        if self._ransac_thresh > 0 and len(good) >= self._min_good:
            pts_cur = np.float32([kp_cur[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(pts_cur, pts_ref, cv2.RANSAC, self._ransac_thresh)
            if mask is not None:
                good = [g for g, m in zip(good, mask.ravel()) if m]

        denom = max(len(kp_cur), len(kp_ref))
        return float(len(good)) / denom if denom > 0 else 0.0

    # ------------------------------------------------------------------
    # C. SSIM Score
    # ------------------------------------------------------------------
    @staticmethod
    def _ssim_score(gray_cur: np.ndarray, gray_ref: np.ndarray) -> float:
        # Images are already the same size (both preprocessed to target_size)
        score, _ = ssim(gray_cur, gray_ref, full=True)
        return max(0.0, float(score))

    # ------------------------------------------------------------------
    # D. Edge Difference
    # ------------------------------------------------------------------
    @staticmethod
    def _edge_difference(edges_cur: np.ndarray, edges_ref: np.ndarray) -> float:
        """Normalised mean absolute difference of edge maps. Lower = more similar."""
        cur_f = edges_cur.astype(np.float32) / 255.0
        ref_f = edges_ref.astype(np.float32) / 255.0
        return float(np.mean(np.abs(cur_f - ref_f)))

    # ------------------------------------------------------------------
    # E. Histogram Difference
    # ------------------------------------------------------------------
    @staticmethod
    def _histogram_difference(hsv_cur: np.ndarray, hsv_ref: np.ndarray) -> float:
        """
        Normalised colour histogram distance in HSV space.
        Uses Bhattacharyya coefficient converted to a difference score.
        Lower = more similar colour distribution.
        """
        h_bins, s_bins = 50, 60
        hist_size = [h_bins, s_bins]
        h_range = [0, 180]
        s_range = [0, 256]
        ranges = h_range + s_range
        channels = [0, 1]   # H and S channels only (ignore V for illumination robustness)

        hist_cur = cv2.calcHist([hsv_cur], channels, None, hist_size, ranges)
        hist_ref = cv2.calcHist([hsv_ref], channels, None, hist_size, ranges)

        cv2.normalize(hist_cur, hist_cur, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_ref, hist_ref, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Bhattacharyya: 0 = identical, 1 = completely different
        diff = cv2.compareHist(hist_cur, hist_ref, cv2.HISTCMP_BHATTACHARYYA)
        return float(np.clip(diff, 0.0, 1.0))

    # ------------------------------------------------------------------
    # F. Shape Difference (Hu Moments)
    # ------------------------------------------------------------------
    @staticmethod
    def _shape_difference(gray_cur: np.ndarray, gray_ref: np.ndarray) -> float:
        """
        Hu-moment distance between the two images.
        Lower = more similar shape.
        """
        def _hu_moments(gray: np.ndarray) -> np.ndarray:
            moments = cv2.moments(gray.astype(np.float64))
            hu = cv2.HuMoments(moments).flatten()
            # Log-transform to normalise the wide range
            with np.errstate(divide="ignore", invalid="ignore"):
                hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
            return hu_log

        hu_cur = _hu_moments(gray_cur)
        hu_ref = _hu_moments(gray_ref)
        diff = float(np.linalg.norm(hu_cur - hu_ref))
        # Normalise by a practical upper bound (~30) to keep in [0,1]
        return float(np.clip(diff / 30.0, 0.0, 1.0))
