"""
SealScan -- Pydantic request / response schemas.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request bodies (base64 JSON)
# ---------------------------------------------------------------------------
class QualityCheckRequest(BaseModel):
    image: str = Field(
        ...,
        description=(
            "Base64-encoded image string. Accepts plain base64 or a data-URI "
            "(e.g. 'data:image/jpeg;base64,...')."
        ),
    )


class SimilarityRequest(BaseModel):
    current_image: str = Field(
        ...,
        description="Base64-encoded current seal photograph.",
    )
    reference_images: list[str] = Field(
        ...,
        min_length=1,
        description="List of base64-encoded reference/previous seal images (at least one required).",
    )



# ---------------------------------------------------------------------------
# Quality metric detail
# ---------------------------------------------------------------------------
class MetricDetail(BaseModel):
    metric: str
    value: float
    threshold: Optional[float] = None
    passed: bool
    message: Optional[str] = None


class ResolutionDetail(BaseModel):
    width: int
    height: int
    passed: bool


class AllMetrics(BaseModel):
    resolution: ResolutionDetail
    sharpness: MetricDetail
    brenner_sharpness: MetricDetail
    canny_edge_density: MetricDetail
    brightness: MetricDetail
    contrast: MetricDetail
    noise: MetricDetail


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------
class ErrorBody(BaseModel):
    code: str
    message: str


# ---------------------------------------------------------------------------
# Quality-check responses
# ---------------------------------------------------------------------------
class QualityCheckResponse(BaseModel):
    success: bool
    quality_passed: bool
    message: str
    metrics: Optional[AllMetrics] = None
    failed_metrics: Optional[list[MetricDetail]] = None
    error: Optional[ErrorBody] = None


# ---------------------------------------------------------------------------
# Per-reference comparison
# ---------------------------------------------------------------------------
class ReferenceComparison(BaseModel):
    reference_id: str
    cosine_similarity: float
    orb_match_ratio: float
    ssim_score: float
    edge_difference: float
    histogram_difference: float
    shape_difference: float


# ---------------------------------------------------------------------------
# Aggregated metrics
# ---------------------------------------------------------------------------
class AggregatedMetrics(BaseModel):
    cosine_similarity: float
    orb_match_ratio: float
    ssim_score: float
    edge_difference: float
    histogram_difference: float
    shape_difference: float


# ---------------------------------------------------------------------------
# Tampering assessment
# ---------------------------------------------------------------------------
class TamperingAssessment(BaseModel):
    tampering_score: float = Field(
        ...,
        description=(
            "Classifier score in [0, 1]. Higher values indicate higher tampering risk. "
            "This is NOT a calibrated probability unless stated otherwise."
        ),
    )
    risk_class: str = Field(
        ...,
        description="Tampering risk class: LOW | MEDIUM | HIGH",
    )


# ---------------------------------------------------------------------------
# Similarity-endpoint responses
# ---------------------------------------------------------------------------
class SimilaritySuccessResponse(BaseModel):
    success: bool = True
    quality_passed: bool = True
    message: str
    tampering_assessment: TamperingAssessment
    aggregated_metrics: AggregatedMetrics
    reference_comparisons: list[ReferenceComparison]


class SimilarityQualityFailResponse(BaseModel):
    success: bool = False
    quality_passed: bool = False
    message: str
    failed_metrics: list[MetricDetail]


class SimilarityErrorResponse(BaseModel):
    success: bool = False
    error: ErrorBody
