"""
SealScan -- POST /quality-check endpoint.

Accepts one image and returns quality validation results.
Does NOT perform any similarity / tampering analysis.
"""
from __future__ import annotations
import logging
import time
import uuid

from fastapi import APIRouter, File, UploadFile, Request

from app.services.image_quality import check_quality
from app.models.schemas import QualityCheckResponse, AllMetrics, ErrorBody
from app.utils.image_utils import validate_upload, bytes_to_bgr

logger = logging.getLogger("sealscan.api.quality")

router = APIRouter()


@router.post(
    "/quality-check",
    response_model=QualityCheckResponse,
    summary="Image Quality Check",
    description=(
        "Validate a single seal photograph against quality thresholds. "
        "Does not perform tampering analysis."
    ),
    tags=["Quality Check"],
)
async def quality_check(image: UploadFile = File(..., description="Seal image to quality-check")):
    request_id = str(uuid.uuid4())[:8]
    t_start = time.perf_counter()
    logger.info("[%s] POST /quality-check | filename=%s", request_id, image.filename)

    try:
        # 1. Validate file
        raw = validate_upload(image)

        # 2. Decode
        bgr = bytes_to_bgr(raw, label="uploaded image")

        # 3. Quality check
        passed, all_metrics, failed_metrics = check_quality(bgr)

        elapsed = round(time.perf_counter() - t_start, 3)

        if passed:
            logger.info("[%s] Quality PASSED in %.3fs", request_id, elapsed)
            return QualityCheckResponse(
                success=True,
                quality_passed=True,
                message="Image quality is satisfactory.",
                metrics=all_metrics,
            )
        else:
            logger.info(
                "[%s] Quality FAILED in %.3fs | failed=%s",
                request_id, elapsed, [m.metric for m in failed_metrics],
            )
            return QualityCheckResponse(
                success=False,
                quality_passed=False,
                message="Image quality is not satisfactory. Please retake the image.",
                failed_metrics=failed_metrics,
            )

    except Exception as exc:
        # HTTPExceptions are re-raised as-is; unexpected exceptions are wrapped
        from fastapi import HTTPException
        if isinstance(exc, HTTPException):
            raise
        logger.exception("[%s] Unexpected error in /quality-check", request_id)
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
