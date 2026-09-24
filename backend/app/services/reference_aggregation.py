"""
SealScan -- Reference Image Aggregation Service.

Aggregates per-reference similarity metrics into a single feature vector
using the strategy defined in settings.REFERENCE_AGGREGATION_METHOD.

This module is intentionally isolated so the aggregation strategy can
be changed without touching feature extraction or the classifier.
"""
from __future__ import annotations
import logging
import statistics
from typing import Literal

import numpy as np

from app.config.settings import (
    REFERENCE_AGGREGATION_METHOD,
    BEST_MATCH_METRIC,
    CLASSIFIER_FEATURE_ORDER,
)

logger = logging.getLogger("sealscan.reference_aggregation")

# Metrics where HIGHER = better similarity
_HIGHER_IS_BETTER = {"cosine_similarity", "orb_match_ratio", "ssim_score"}
# Metrics where LOWER = better similarity
_LOWER_IS_BETTER = {"edge_difference", "histogram_difference", "shape_difference"}


def aggregate(
    reference_comparisons: list[dict],
    method: str | None = None,
) -> dict:
    """
    Aggregate a list of per-reference metric dicts into one aggregated metric dict.

    Args:
        reference_comparisons: list of dicts with keys = metric names
        method: override the global setting (useful for testing)

    Returns:
        dict with the same keys as CLASSIFIER_FEATURE_ORDER
    """
    if not reference_comparisons:
        raise ValueError("Cannot aggregate an empty reference list.")

    strategy = (method or REFERENCE_AGGREGATION_METHOD).lower()

    if strategy == "best_match":
        return _aggregate_best_match(reference_comparisons)
    elif strategy == "mean":
        return _aggregate_mean(reference_comparisons)
    elif strategy == "median":
        return _aggregate_median(reference_comparisons)
    else:
        logger.warning("Unknown aggregation method '%s'; falling back to best_match.", strategy)
        return _aggregate_best_match(reference_comparisons)


def _aggregate_best_match(comparisons: list[dict]) -> dict:
    """
    Select the reference image that is most similar to the current image,
    based on BEST_MATCH_METRIC, and return its metrics as the aggregated result.
    """
    metric = BEST_MATCH_METRIC
    if metric in _HIGHER_IS_BETTER:
        best = max(comparisons, key=lambda d: d[metric])
    else:
        best = min(comparisons, key=lambda d: d[metric])

    logger.info("best_match | selected reference=%s via metric=%s", best.get("reference_id"), metric)
    return {k: best[k] for k in CLASSIFIER_FEATURE_ORDER}


def _aggregate_mean(comparisons: list[dict]) -> dict:
    return {
        metric: round(float(np.mean([d[metric] for d in comparisons])), 6)
        for metric in CLASSIFIER_FEATURE_ORDER
    }


def _aggregate_median(comparisons: list[dict]) -> dict:
    return {
        metric: round(float(np.median([d[metric] for d in comparisons])), 6)
        for metric in CLASSIFIER_FEATURE_ORDER
    }
