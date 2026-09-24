"""
SealScan -- POST /seal-scan/similarity endpoint.

Full pipeline: quality check -> preprocessing -> feature extraction
-> reference aggregation -> classifier -> response.
"""
from __future__ import annotations
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, status

from app.services.image_quality import check_quality
from app.services.similarity_engine import SimilarityEngine
from app.services.reference_aggregation import aggregate
from app.models.schemas import (
    SimilarityRequest,
    SimilaritySuccessResponse,
    SimilarityQualityFailResponse,
    TamperingAssessment,
    AggregatedMetrics,
    ReferenceComparison,
)
from app.utils.image_utils import base64_to_bgr

logger = logging.getLogger("sealscan.api.similarity")

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level singletons (initialised at import time so the classifier
# model is loaded once when the application starts).
# ---------------------------------------------------------------------------
_similarity_engine: SimilarityEngine | None = None
_classifier = None


def _get_engine() -> SimilarityEngine:
    global _similarity_engine
    if _similarity_engine is None:
        _similarity_engine = SimilarityEngine()
    return _similarity_engine


def _get_classifier():
    global _classifier
    if _classifier is None:
        from app.services.tampering_classifier import TamperingClassifier
        _classifier = TamperingClassifier()
        logger.info("TamperingClassifier ready (mode=%s)", _classifier.mode)
    return _classifier


@router.post(
    "/seal-scan/similarity",
    summary="Seal Similarity & Tampering Assessment",
    description=(
        "Upload a current seal image and one or more reference images. "
        "The backend validates image quality, extracts similarity features, "
        "and returns a tampering risk assessment (LOW / MEDIUM / HIGH). "
        "This is an AI-assisted risk assessment tool; results are decision-support "
        "only and do not constitute a legal determination."
    ),
    tags=["Similarity & Tampering"],
)
async def seal_similarity(body: SimilarityRequest):
    request_id = str(uuid.uuid4())[:8]
    t_start = time.perf_counter()
    logger.info(
        "[%s] POST /seal-scan/similarity | n_references=%d",
        request_id, len(body.reference_images),
    )

    try:
        # ----------------------------------------------------------------
        # 1. Decode current image (base64 → BGR)
        # ----------------------------------------------------------------
        cur_bgr = base64_to_bgr(body.current_image, label="current_image")

        # ----------------------------------------------------------------
        # 3. Image quality check on current image only
        # ----------------------------------------------------------------
        quality_passed, all_metrics, failed_metrics = check_quality(cur_bgr)

        if not quality_passed:
            elapsed = round(time.perf_counter() - t_start, 3)
            logger.info(
                "[%s] Quality FAILED in %.3fs | failed=%s",
                request_id, elapsed, [m.metric for m in failed_metrics],
            )
            return SimilarityQualityFailResponse(
                success=False,
                quality_passed=False,
                message="Image quality is not satisfactory. Please retake the image.",
                failed_metrics=failed_metrics,
            )

        # ----------------------------------------------------------------
        # 4. Decode reference images (base64 → BGR)
        # ----------------------------------------------------------------
        ref_bgrs = []
        ref_ids = []
        for i, ref_b64 in enumerate(body.reference_images):
            ref_id = f"reference_{i + 1}"
            ref_bgr = base64_to_bgr(ref_b64, label=f"reference image {i + 1}")
            ref_bgrs.append(ref_bgr)
            ref_ids.append(ref_id)

        # ----------------------------------------------------------------
        # 5. Preprocessing + Feature Extraction (via Similarity Engine)
        # ----------------------------------------------------------------
        engine = _get_engine()
        reference_comparisons_raw = engine.compare(cur_bgr, ref_bgrs, ref_ids)

        # ----------------------------------------------------------------
        # 6. Reference Aggregation
        # ----------------------------------------------------------------
        aggregated_raw = aggregate(reference_comparisons_raw)

        # ----------------------------------------------------------------
        # 7. Tampering Classification
        # ----------------------------------------------------------------
        classifier = _get_classifier()
        tampering_score, risk_class = classifier.predict(aggregated_raw)

        elapsed = round(time.perf_counter() - t_start, 3)
        logger.info(
            "[%s] Analysis completed in %.3fs | n_refs=%d | risk=%s | score=%.4f",
            request_id, elapsed, len(ref_bgrs), risk_class, tampering_score,
        )

        # ----------------------------------------------------------------
        # 8. Build response
        # ----------------------------------------------------------------
        return SimilaritySuccessResponse(
            success=True,
            quality_passed=True,
            message=(
                "Image quality satisfactory. Similarity analysis completed. "
                "Tampering risk assessment: "
                + risk_class
                + ". This is an AI-assisted assessment; the inspecting officer''s "
                  "determination is authoritative."
            ),
            tampering_assessment=TamperingAssessment(
                tampering_score=tampering_score,
                risk_class=risk_class,
            ),
            aggregated_metrics=AggregatedMetrics(**aggregated_raw),
            reference_comparisons=[
                ReferenceComparison(**r) for r in reference_comparisons_raw
            ],
        )

    except Exception as exc:
        from fastapi import HTTPException as _HTTPException
        if isinstance(exc, _HTTPException):
            raise
        logger.exception("[%s] Unexpected error in /seal-scan/similarity", request_id)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred. Please try again.",
                },
            },
        ) from exc
