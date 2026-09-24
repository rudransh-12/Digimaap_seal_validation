# SealScan Backend

> **AI-Assisted Legal Metrology Seal Verification System**
>
> _All tampering risk assessments are decision-support tools only.
> They do not constitute a legal determination.
> The inspecting officer''s final judgment is authoritative._

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Generate the Demo Classifier](#generate-the-demo-classifier)
5. [Configuration](#configuration)
6. [API Endpoints](#api-endpoints)
   - [POST /quality-check](#post-quality-check)
   - [POST /seal-scan/similarity](#post-seal-scansimilarity)
7. [Similarity Metrics Reference](#similarity-metrics-reference)
8. [Tampering Score Reference](#tampering-score-reference)
9. [Quality Metric Thresholds](#quality-metric-thresholds)
10. [Error Codes](#error-codes)
11. [Running Tests](#running-tests)
12. [Flutter Integration Guide](#flutter-integration-guide)
13. [Example curl Requests](#example-curl-requests)
14. [Example JSON Responses](#example-json-responses)
15. [Project Structure](#project-structure)

---

## Architecture Overview

```
Flutter Client
       |
       |  multipart/form-data
       v
+----------------------------------------------+
|              FastAPI Application              |
|  POST /quality-check                         |
|  POST /seal-scan/similarity                  |
+----------------------------------------------+
       |
       v
+----------------------------------------------+
|           API Layer (thin routers)           |
+----------------------------------------------+
       |
       v
+----------------------------------------------+
|         File Validation (image_utils)        |
+----------------------------------------------+
       |
       v
+----------------------------------------------+
|         Image Quality Service                |
|   - Resolution check                        |
|   - Sharpness (Laplacian variance)          |
|   - Brightness (mean pixel)                 |
|   - Contrast (std dev)                      |
|   - Noise estimate                          |
+----------------------------------------------+
   FAIL |              | PASS
        v              v
  Return failed    Preprocessing
  metrics          (resize, denoise,
                    grayscale, HSV,
                    edge map)
                        |
                        v
               Feature Extraction
               - Cosine Similarity
               - ORB Match Ratio (+ RANSAC)
               - SSIM Score
               - Edge Difference
               - Histogram Difference (HSV)
               - Shape Difference (Hu Moments)
                        |
                        v
               Reference Aggregation
               (best_match / mean / median)
                        |
                        v
               Tampering Classifier
               (.pkl or heuristic fallback)
                        |
                        v
               JSON Response
```

---

## Prerequisites

- Python 3.10+
- pip

---

## Quick Start

```bash
# 1. Navigate to the backend directory
cd backend

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional but recommended) Generate the demo classifier
python model/generate_demo_model.py

# 5. Start the development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be live at: http://localhost:8000
Interactive docs: http://localhost:8000/docs
OpenAPI JSON: http://localhost:8000/openapi.json

---

## Generate the Demo Classifier

The system works without a trained model (heuristic fallback is used automatically),
but for real deployments train and place the model at `model/tampering_classifier.pkl`.

To generate a synthetic demo model:

```bash
python model/generate_demo_model.py
```

This creates `model/tampering_classifier.pkl` using 1500 synthetic training samples.

To replace with a real model:
1. Train a sklearn-compatible classifier on real seal comparison data.
2. Ensure your training features are in the same order as `CLASSIFIER_FEATURE_ORDER`
   in `app/config/settings.py`.
3. Save with `pickle.dump(model, open("model/tampering_classifier.pkl", "wb"))`.
4. Restart the server — the model is loaded once at startup.

---

## Configuration

All tunable parameters are in `app/config/settings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_IMAGE_SIZE_BYTES` | 20 MB | Maximum upload file size |
| `QUALITY_THRESHOLDS["min_width"]` | 320 | Minimum image width (px) |
| `QUALITY_THRESHOLDS["min_height"]` | 320 | Minimum image height (px) |
| `QUALITY_THRESHOLDS["min_sharpness"]` | 80.0 | Minimum Laplacian variance |
| `QUALITY_THRESHOLDS["min_brightness"]` | 30.0 | Minimum mean pixel value |
| `QUALITY_THRESHOLDS["max_brightness"]` | 225.0 | Maximum mean pixel value |
| `QUALITY_THRESHOLDS["min_contrast"]` | 20.0 | Minimum std-dev of pixel values |
| `QUALITY_THRESHOLDS["max_noise"]` | 15.0 | Maximum noise std-dev |
| `PREPROCESSING["target_size"]` | (512, 512) | Working resolution for CV |
| `REFERENCE_AGGREGATION_METHOD` | `best_match` | How multiple references are aggregated |
| `BEST_MATCH_METRIC` | `ssim_score` | Metric used to select the best reference |

---

## API Endpoints

### POST /quality-check

**Purpose:** Validate image quality only. No similarity or tampering analysis is performed.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | Yes | Seal photograph (JPEG, PNG, BMP, TIFF, WebP) |

**Success Response (200):**
```json
{
  "success": true,
  "quality_passed": true,
  "message": "Image quality is satisfactory.",
  "metrics": {
    "resolution": { "width": 1920, "height": 1080, "passed": true },
    "sharpness": { "metric": "sharpness", "value": 245.62, "threshold": 80.0, "passed": true },
    "brightness": { "metric": "brightness", "value": 128.4, "threshold": 225.0, "passed": true },
    "contrast": { "metric": "contrast", "value": 57.2, "threshold": 20.0, "passed": true },
    "noise": { "metric": "noise", "value": 3.1, "threshold": 15.0, "passed": true }
  }
}
```

**Failure Response (200):**
```json
{
  "success": false,
  "quality_passed": false,
  "message": "Image quality is not satisfactory. Please retake the image.",
  "failed_metrics": [
    {
      "metric": "sharpness",
      "value": 38.5,
      "threshold": 80.0,
      "passed": false,
      "message": "Image is too blurry. Please retake with a steadier hand."
    }
  ]
}
```

---

### POST /seal-scan/similarity

**Purpose:** Full tampering assessment pipeline — quality check, feature extraction, and classifier.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `current_image` | file | Yes | Current seal photograph |
| `reference_images` | file[] | Yes (>=1) | Previously verified reference seal images |

**Success Response (200):**
```json
{
  "success": true,
  "quality_passed": true,
  "message": "Image quality satisfactory. Similarity analysis completed. Tampering risk assessment: LOW. ...",
  "tampering_assessment": {
    "tampering_score": 0.12,
    "risk_class": "LOW"
  },
  "aggregated_metrics": {
    "cosine_similarity": 0.97,
    "orb_match_ratio": 0.81,
    "ssim_score": 0.94,
    "edge_difference": 0.04,
    "histogram_difference": 0.06,
    "shape_difference": 0.02
  },
  "reference_comparisons": [
    {
      "reference_id": "reference_1",
      "cosine_similarity": 0.97,
      "orb_match_ratio": 0.81,
      "ssim_score": 0.94,
      "edge_difference": 0.04,
      "histogram_difference": 0.06,
      "shape_difference": 0.02
    }
  ]
}
```

**Quality Failure Response (200):**
```json
{
  "success": false,
  "quality_passed": false,
  "message": "Image quality is not satisfactory. Please retake the image.",
  "failed_metrics": [
    {
      "metric": "sharpness",
      "value": 42.17,
      "threshold": 80.0,
      "passed": false,
      "message": "Image is too blurry. Please retake with a steadier hand."
    }
  ]
}
```

---

## Similarity Metrics Reference

| Metric | Range | Better When | Description |
|--------|-------|-------------|-------------|
| `cosine_similarity` | [0, 1] | Higher | Vector-space similarity of flattened greyscale pixel arrays |
| `orb_match_ratio` | [0, 1] | Higher | Ratio of geometrically verified ORB keypoint matches to total keypoints |
| `ssim_score` | [0, 1] | Higher | Structural Similarity Index — measures luminance, contrast, structure |
| `edge_difference` | [0, 1] | Lower | Mean absolute difference of Canny edge maps |
| `histogram_difference` | [0, 1] | Lower | Bhattacharyya distance of normalised HSV colour histograms |
| `shape_difference` | [0, 1] | Lower | Log-normalised Hu-moment distance |

---

## Tampering Score Reference

`tampering_score` is a float in [0, 1] produced by the classifier.

- **0.0** — metrics indicate the images are virtually identical
- **1.0** — metrics indicate highly dissimilar images (high tampering risk)

**Important:** Unless the deployed model has been explicitly calibrated
(e.g., via `CalibratedClassifierCV`), the score is NOT a statistical probability.
It is a risk indicator derived from the classifier''s output.

| `risk_class` | Interpretation |
|---|---|
| `LOW` | Similarity metrics suggest the seal is consistent with the reference. |
| `MEDIUM` | Moderate differences detected. Recommend officer review. |
| `HIGH` | Significant differences detected. Seal should be physically inspected. |

---

## Quality Metric Thresholds

| Metric | Method | Pass Condition |
|--------|--------|----------------|
| Resolution | Pixel dimensions | width >= 320 AND height >= 320 |
| Sharpness | Laplacian variance | value >= 80.0 |
| Brightness | Mean greyscale | 30.0 <= value <= 225.0 |
| Contrast | Std-dev greyscale | value >= 20.0 |
| Noise | Blur residual std | value <= 15.0 |

All thresholds are configurable in `app/config/settings.py`.

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_FILE_TYPE` | 422 | Unsupported file extension |
| `EMPTY_FILE` | 422 | Uploaded file has zero bytes |
| `FILE_TOO_LARGE` | 413 | File exceeds 20 MB limit |
| `INVALID_IMAGE` | 422 | File cannot be decoded as an image |
| `MISSING_REFERENCE_IMAGES` | 422 | No reference images provided |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

Test files:
- `tests/test_quality.py` — Image quality service unit tests
- `tests/test_similarity.py` — Feature extraction, aggregation, classifier unit tests
- `tests/test_api.py` — End-to-end FastAPI integration tests

---

## Flutter Integration Guide

### Quality Check

```dart
Future<Map<String, dynamic>> checkQuality(File imageFile) async {
  final request = http.MultipartRequest(
    "POST",
    Uri.parse("http://<server>:8000/quality-check"),
  );
  request.files.add(
    await http.MultipartFile.fromPath("image", imageFile.path),
  );
  final response = await request.send();
  final body = await response.stream.bytesToString();
  return json.decode(body);
}
```

**Response handling:**
```dart
if (result["quality_passed"] == true) {
  // Proceed with inspection
} else {
  final failedMetrics = result["failed_metrics"] as List;
  for (final m in failedMetrics) {
    showError(m["message"]); // e.g. "Image is too blurry"
  }
}
```

### Similarity & Tampering Assessment

```dart
Future<Map<String, dynamic>> assessTampering(
  File currentImage,
  List<File> referenceImages,
) async {
  final request = http.MultipartRequest(
    "POST",
    Uri.parse("http://<server>:8000/seal-scan/similarity"),
  );

  request.files.add(
    await http.MultipartFile.fromPath("current_image", currentImage.path),
  );

  for (final ref in referenceImages) {
    request.files.add(
      await http.MultipartFile.fromPath("reference_images", ref.path),
    );
  }

  final response = await request.send();
  final body = await response.stream.bytesToString();
  return json.decode(body);
}
```

**Response handling:**
```dart
if (result["quality_passed"] == false) {
  showRetakePrompt(result["failed_metrics"]);
  return;
}

final assessment = result["tampering_assessment"];
final riskClass = assessment["risk_class"];   // "LOW" | "MEDIUM" | "HIGH"
final score = assessment["tampering_score"];  // 0.0 - 1.0

showRiskBadge(riskClass, score);
// IMPORTANT: Always show the disclaimer:
// "AI-Assisted Tampering Risk Assessment. Officer determination is authoritative."
```

---

## Example curl Requests

### Quality Check

```bash
curl -X POST http://localhost:8000/quality-check \
  -F "image=@/path/to/seal.jpg"
```

### Similarity + Tampering Assessment (single reference)

```bash
curl -X POST http://localhost:8000/seal-scan/similarity \
  -F "current_image=@/path/to/current.jpg" \
  -F "reference_images=@/path/to/reference1.jpg"
```

### Similarity + Tampering Assessment (multiple references)

```bash
curl -X POST http://localhost:8000/seal-scan/similarity \
  -F "current_image=@/path/to/current.jpg" \
  -F "reference_images=@/path/to/reference1.jpg" \
  -F "reference_images=@/path/to/reference2.jpg" \
  -F "reference_images=@/path/to/reference3.jpg"
```

---

## Example JSON Responses

### Quality Check — All Passing

```json
{
  "success": true,
  "quality_passed": true,
  "message": "Image quality is satisfactory.",
  "metrics": {
    "resolution": { "width": 1920, "height": 1080, "passed": true },
    "sharpness":  { "metric": "sharpness",  "value": 312.4, "threshold": 80.0,  "passed": true },
    "brightness": { "metric": "brightness", "value": 130.1, "threshold": 225.0, "passed": true },
    "contrast":   { "metric": "contrast",   "value": 62.3,  "threshold": 20.0,  "passed": true },
    "noise":      { "metric": "noise",      "value": 4.2,   "threshold": 15.0,  "passed": true }
  }
}
```

### Quality Check — Blurry Image

```json
{
  "success": false,
  "quality_passed": false,
  "message": "Image quality is not satisfactory. Please retake the image.",
  "failed_metrics": [
    {
      "metric": "sharpness",
      "value": 38.5,
      "threshold": 80.0,
      "passed": false,
      "message": "Image is too blurry. Please retake with a steadier hand."
    }
  ]
}
```

### Similarity — LOW Risk

```json
{
  "success": true,
  "quality_passed": true,
  "message": "Image quality satisfactory. Similarity analysis completed. Tampering risk assessment: LOW. This is an AI-assisted assessment; the inspecting officer''s determination is authoritative.",
  "tampering_assessment": {
    "tampering_score": 0.09,
    "risk_class": "LOW"
  },
  "aggregated_metrics": {
    "cosine_similarity": 0.978,
    "orb_match_ratio": 0.843,
    "ssim_score": 0.961,
    "edge_difference": 0.023,
    "histogram_difference": 0.041,
    "shape_difference": 0.011
  },
  "reference_comparisons": [
    {
      "reference_id": "reference_1",
      "cosine_similarity": 0.978,
      "orb_match_ratio": 0.843,
      "ssim_score": 0.961,
      "edge_difference": 0.023,
      "histogram_difference": 0.041,
      "shape_difference": 0.011
    }
  ]
}
```

### Similarity — HIGH Risk

```json
{
  "success": true,
  "quality_passed": true,
  "message": "Image quality satisfactory. Similarity analysis completed. Tampering risk assessment: HIGH. This is an AI-assisted assessment; the inspecting officer''s determination is authoritative.",
  "tampering_assessment": {
    "tampering_score": 0.89,
    "risk_class": "HIGH"
  },
  "aggregated_metrics": {
    "cosine_similarity": 0.21,
    "orb_match_ratio": 0.08,
    "ssim_score": 0.14,
    "edge_difference": 0.72,
    "histogram_difference": 0.81,
    "shape_difference": 0.68
  },
  "reference_comparisons": [
    {
      "reference_id": "reference_1",
      "cosine_similarity": 0.31,
      "orb_match_ratio": 0.12,
      "ssim_score": 0.22,
      "edge_difference": 0.61,
      "histogram_difference": 0.70,
      "shape_difference": 0.55
    },
    {
      "reference_id": "reference_2",
      "cosine_similarity": 0.21,
      "orb_match_ratio": 0.08,
      "ssim_score": 0.14,
      "edge_difference": 0.72,
      "histogram_difference": 0.81,
      "shape_difference": 0.68
    }
  ]
}
```

### Error — Invalid File Type

```json
{
  "success": false,
  "error": {
    "code": "INVALID_FILE_TYPE",
    "message": "Unsupported file type '.pdf'. Allowed types: .jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp"
  }
}
```

---

## Project Structure

```
backend/
 app/
   config/
     settings.py          # All tunable thresholds and parameters
   api/
     quality.py           # POST /quality-check route
     similarity.py        # POST /seal-scan/similarity route
   services/
     image_quality.py     # Quality metrics evaluation
     preprocessing.py     # Image preprocessing pipeline
     feature_extraction.py # Six similarity metrics
     similarity_engine.py # Orchestrates preprocessing + extraction
     reference_aggregation.py # best_match / mean / median strategies
     tampering_classifier.py  # Model loader + heuristic fallback
   models/
     schemas.py           # Pydantic request/response models
   utils/
     image_utils.py       # File validation, decode, colour conversions
   main.py                # FastAPI app, middleware, startup warmup
 model/
   generate_demo_model.py # Script to produce tampering_classifier.pkl
   tampering_classifier.pkl  # (generated) trained sklearn model
 tests/
   test_quality.py        # Quality service unit tests
   test_similarity.py     # Feature / aggregation / classifier unit tests
   test_api.py            # FastAPI integration tests
 requirements.txt
 .env.example
 README.md
```
