"""
SealScan -- Similarity Engine.

Orchestrates preprocessing + feature extraction over the current image
and all reference images, then returns per-reference metrics.
"""
from __future__ import annotations
import logging

import numpy as np

from app.services.preprocessing import ImagePreprocessor
from app.services.feature_extraction import FeatureExtractor, SimilarityMetrics

logger = logging.getLogger("sealscan.similarity_engine")


class SimilarityEngine:
    """Processes one current image against N reference BGR arrays."""

    def __init__(self) -> None:
        self._preprocessor = ImagePreprocessor()
        self._extractor = FeatureExtractor()

    def compare(
        self,
        current_bgr: np.ndarray,
        reference_bgrs: list[np.ndarray],
        reference_ids: list[str],
    ) -> list[dict]:
        """
        Run preprocessing + metric extraction for every reference image.

        Returns a list of dicts (one per reference) containing reference_id
        plus all six metric values.
        """
        logger.info("Preprocessing current image...")
        current_prep = self._preprocessor.process(current_bgr)

        results: list[dict] = []
        for idx, (ref_bgr, ref_id) in enumerate(zip(reference_bgrs, reference_ids)):
            logger.info("Comparing against reference %d/%d [%s]", idx + 1, len(reference_bgrs), ref_id)
            ref_prep = self._preprocessor.process(ref_bgr)
            metrics: SimilarityMetrics = self._extractor.compute(current_prep, ref_prep)
            results.append(
                {
                    "reference_id": ref_id,
                    "cosine_similarity": metrics.cosine_similarity,
                    "orb_match_ratio": metrics.orb_match_ratio,
                    "ssim_score": metrics.ssim_score,
                    "edge_difference": metrics.edge_difference,
                    "histogram_difference": metrics.histogram_difference,
                    "shape_difference": metrics.shape_difference,
                }
            )

        return results
