# SealScan Backend

> **AI-Assisted Legal Metrology Seal Verification System**
>
> _All tampering risk assessments are decision-support tools only.
> They do not constitute a legal determination.
> The inspecting officer's final judgment is authoritative._

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Trained Models](#trained-models)
5. [Configuration](#configuration)
6. [API Endpoints](#api-endpoints)
   - [POST /quality-check](#post-quality-check)
   - [POST /seal-scan/similarity](#post-seal-scansimilarity)
7. [Image Quality Metrics](#image-quality-metrics)
8. [Similarity Metrics Reference](#similarity-metrics-reference)
9. [Tampering Score Reference](#tampering-score-reference)
10. [Error Codes](#error-codes)
11. [Running Tests](#running-tests)
12. [Flutter Integration Guide](#flutter-integration-guide)
13. [Example curl Requests](#example-curl-requests)
14. [Example JSON Responses](#example-json-responses)
15. [Project Structure](#project-structure)

---

## Architecture Overview

```
Flutter Client (Base64 JSON Payloads)
       |
       |  POST /quality-check       POST /seal-scan/similarity
       v
+-------------------------------------------------------------+
|                     FastAPI Application                     |
|  POST /quality-check          POST /seal-scan/similarity    |
+-------------------------------------------------------------+
       |
       v
+-------------------------------------------------------------+
|                  API Layer (Thin Routers)                   |
+-------------------------------------------------------------+
       |
       v
+-------------------------------------------------------------+
|         Base64 Decoding & Validation (image_utils)          |
+-------------------------------------------------------------+
       |
       v
+-------------------------------------------------------------+
|                    Image Quality Service                    |
|   - Resolution check (min 320x320)                          |
|   - Laplacian Variance Sharpness                            |
|   - Brenner Focus Sharpness                                 |
|   - Canny Edge Density                                      |
|   - AI Blur Classifier (RandomForest ML)                    |
|   - Brightness (mean pixel)                                 |
|   - Contrast (std dev)                                      |
|   - Noise estimate (Gaussian residual)                      |
+-------------------------------------------------------------+
   FAIL |                                       | PASS
        v                                       v
  Return failed                           Preprocessing
  metrics response                        (resize 512x512, denoise,
                                           grayscale, HSV, Canny edges)
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

# 4. Start the development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be live at: http://localhost:8000
Interactive Swagger docs: http://localhost:8000/docs
OpenAPI JSON: http://localhost:8000/openapi.json

---

## Trained Models

The backend utilizes two machine learning models located in `model/`:

1. **Blur Classifier (`model/blur_classifier.pkl`)**:
   - Random Forest binary classifier (150 trees, max depth 6) for image sharpness verification.
   - Evaluates 7 features: Laplacian variance, Sobel edge strength, Gaussian noise residual, Brenner horizontal second-difference, Canny edge density (100, 200), 2D FFT spectral energy ratio, and `skimage.measure.blur_effect`.
   - Loaded once at startup in `app/services/blur_classifier.py`.

2. **Tampering Classifier (`model/tampering_classifier.pkl`)**:
   - Random Forest multi-class risk classifier (`LOW`, `MEDIUM`, `HIGH`) for seal tampering detection.
   - Can be regenerated or demo-trained using `python model/generate_demo_model.py`.
   - Loaded once at startup in `app/services/tampering_classifier.py`.

---

## Configuration

All tunable thresholds and paths are centralized in `app/config/settings.py`:

| Parameter | Default | Description |
|---|---|---|
| `MAX_IMAGE_SIZE_BYTES` | 20 MB | Maximum upload payload size |
| `QUALITY_THRESHOLDS["min_width"]` | 320 | Minimum image width (px) |
| `QUALITY_THRESHOLDS["min_height"]` | 320 | Minimum image height (px) |
| `QUALITY_THRESHOLDS["min_sharpness"]` | 80.0 | Minimum Laplacian variance |
| `QUALITY_THRESHOLDS["min_brenner_sharpness"]` | 100.0 | Minimum Brenner gradient focus measure |
| `QUALITY_THRESHOLDS["min_canny_edge_density"]` | 0.008 | Minimum ratio of edge pixels (0.0 - 1.0) |
| `QUALITY_THRESHOLDS["min_blur_classifier_score"]` | 0.50 | Minimum AI probability of being `NOT_BLURRY` |
| `QUALITY_THRESHOLDS["min_brightness"]` | 30.0 | Minimum mean pixel value |
| `QUALITY_THRESHOLDS["max_brightness"]` | 225.0 | Maximum mean pixel value |
| `QUALITY_THRESHOLDS["min_contrast"]` | 25.0 | Minimum std-dev of pixel values |
| `QUALITY_THRESHOLDS["max_noise"]` | 15.0 | Maximum noise residual std-dev |
| `PREPROCESSING["target_size"]` | (512, 512) | Working resolution for CV operations |
| `REFERENCE_AGGREGATION_METHOD` | `best_match` | Aggregation method across multiple reference images |
| `BEST_MATCH_METRIC` | `ssim_score` | Metric used to select best reference image |

---

## API Endpoints

### POST /quality-check

**Purpose:** Rapidly validate image quality without running the full comparison pipeline.

**Request:** `Content-Type: application/json`
```json
{
  "image": "<base64 encoded image string or data:image/...;base64,...>"
}
```

**Success Response (200 OK):**
```json
{
  "success": true,
  "quality_passed": true,
  "message": "Image quality is satisfactory.",
  "metrics": {
    "resolution":        { "width": 1920, "height": 1080, "passed": true },
    "sharpness":         { "metric": "sharpness",          "value": 245.62, "threshold": 80.0,   "passed": true, "message": null },
    "brenner_sharpness": { "metric": "brenner_sharpness",  "value": 412.50, "threshold": 100.0,  "passed": true, "message": null },
    "canny_edge_density":{ "metric": "canny_edge_density", "value": 0.0452, "threshold": 0.008,  "passed": true, "message": null },
    "blur_classifier":   { "metric": "blur_classifier",   "value": 0.9820, "threshold": 0.50,   "passed": true, "message": null },
    "brightness":        { "metric": "brightness",        "value": 128.40, "threshold": 225.0,  "passed": true, "message": null },
    "contrast":          { "metric": "contrast",          "value": 57.20,  "threshold": 25.0,   "passed": true, "message": null },
    "noise":             { "metric": "noise",             "value": 3.10,   "threshold": 15.0,   "passed": true, "message": null }
  }
}
```

**Failure Response (200 OK):**
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
    },
    {
      "metric": "blur_classifier",
      "value": 0.1240,
      "threshold": 0.50,
      "passed": false,
      "message": "Image failed AI blur assessment. Please ensure camera is focused and steady."
    }
  ]
}
```

---

### POST /seal-scan/similarity

**Purpose:** Full seal verification pipeline (Quality check -> Preprocessing -> Pairwise feature extraction -> Reference aggregation -> Tampering classification).

**Request:** `Content-Type: application/json`
```json
{
  "current_image": "<base64 encoded image string>",
  "reference_images": [
    "<base64 reference image 1>",
    "<base64 reference image 2>"
  ]
}
```

**Success Response (200 OK):**
```json
{
  "success": true,
  "quality_passed": true,
  "message": "Image quality satisfactory. Similarity analysis completed. Tampering risk assessment: LOW. This is an AI-assisted assessment; the inspecting officer's determination is authoritative.",
  "tampering_assessment": {
    "tampering_score": 0.12,
    "risk_class": "LOW"
  },
  "aggregated_metrics": {
    "cosine_similarity":    0.97,
    "orb_match_ratio":      0.81,
    "ssim_score":           0.94,
    "edge_difference":      0.04,
    "histogram_difference": 0.06,
    "shape_difference":     0.02
  },
  "reference_comparisons": [
    {
      "reference_id":         "reference_1",
      "cosine_similarity":    0.97,
      "orb_match_ratio":      0.81,
      "ssim_score":           0.94,
      "edge_difference":      0.04,
      "histogram_difference": 0.06,
      "shape_difference":     0.02
    }
  ]
}
```

**Quality Failure Response (200 OK):**
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

## Image Quality Metrics

| Metric | Method | Pass Condition | Description |
|---|---|---|---|
| `resolution` | Pixel dimensions | width >= 320 AND height >= 320 | Minimum resolution required |
| `sharpness` | Laplacian variance | value >= 80.0 | High-frequency edge gradient variance |
| `brenner_sharpness` | Brenner horizontal second-diff | value >= 100.0 | Step-2 squared intensity differential focus |
| `canny_edge_density` | Canny edge pixel ratio | value >= 0.008 | Ensures presence of fine seal engravings |
| `blur_classifier` | Pretrained Random Forest | value >= 0.50 | Probability of image being `NOT_BLURRY` |
| `brightness` | Greyscale mean pixel | 30.0 <= value <= 225.0 | Checks underexposure / overexposure |
| `contrast` | Greyscale std-dev | value >= 25.0 | Checks dynamic range of the seal |
| `noise` | Gaussian blur residual std | value <= 15.0 | High-frequency sensor noise detection |

---

## Similarity Metrics Reference

| Metric | Range | Better When | Description |
|---|---|---|---|
| `cosine_similarity` | [0, 1] | Higher | Normalised greyscale dot product similarity |
| `orb_match_ratio` | [0, 1] | Higher | Geometrically verified ORB keypoints (RANSAC homography) |
| `ssim_score` | [0, 1] | Higher | Structural Similarity Index (luminance, contrast, structure) |
| `edge_difference` | [0, 1] | Lower | Mean absolute difference of Otsu-Canny edge maps |
| `histogram_difference` | [0, 1] | Lower | Bhattacharyya distance of normalised HSV colour histograms |
| `shape_difference` | [0, 1] | Lower | Log-normalised Hu-moment distance |

---

## Tampering Score Reference

`tampering_score` is a value in `[0, 1]` indicating the estimated risk of seal modification:

* **0.0 - 0.35 (`LOW`)**: Metrics indicate the seal matches known reference specifications.
* **0.36 - 0.70 (`MEDIUM`)**: Moderate differences detected; officer review advised.
* **0.71 - 1.00 (`HIGH`)**: Substantial discrepancies detected; manual physical inspection recommended.

---

## Error Codes

| Code | HTTP Status | Description |
|---|---|---|
| `INVALID_FILE_TYPE` | 422 | Unsupported format |
| `EMPTY_FILE` | 422 | Empty base64 payload |
| `FILE_TOO_LARGE` | 413 | Image exceeds 20 MB limit |
| `INVALID_IMAGE` | 422 | Base64 string cannot be decoded as a valid image |
| `MISSING_REFERENCE_IMAGES` | 422 | `reference_images` array is empty or missing |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

---

## Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

Test coverage includes:
- `tests/test_quality.py` — Quality metric computations, Brenner focus, Canny density, and ML Blur Classifier
- `tests/test_similarity.py` — Feature extraction, aggregation, tampering classifier inference
- `tests/test_api.py` — End-to-end FastAPI integration testing for both endpoints

---

## Flutter Integration Guide

### Quality Check

```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> checkQuality(File imageFile) async {
  final bytes = await imageFile.readAsBytes();
  final base64String = base64Encode(bytes);

  final response = await http.post(
    Uri.parse('http://<server-ip>:8000/quality-check'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({'image': base64String}),
  );

  return jsonDecode(response.body);
}
```

### Similarity & Tampering Assessment

```dart
Future<Map<String, dynamic>> assessTampering(
  File currentImageFile,
  List<File> referenceImageFiles,
) async {
  final currentB64 = base64Encode(await currentImageFile.readAsBytes());
  final refB64List = <String>[];
  for (final ref in referenceImageFiles) {
    refB64List.add(base64Encode(await ref.readAsBytes()));
  }

  final response = await http.post(
    Uri.parse('http://<server-ip>:8000/seal-scan/similarity'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'current_image': currentB64,
      'reference_images': refB64List,
    }),
  );

  return jsonDecode(response.body);
}
```

---

## Example curl Requests

### Quality Check

```bash
curl -X POST "http://localhost:8000/quality-check" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64_string>"
  }'
```

### Similarity Assessment

```bash
curl -X POST "http://localhost:8000/seal-scan/similarity" \
  -H "Content-Type: application/json" \
  -d '{
    "current_image": "<base64_string_current>",
    "reference_images": [
      "<base64_string_ref_1>",
      "<base64_string_ref_2>"
    ]
  }'
```

---

## Project Structure

```
backend/
 ├── app/
 │    ├── config/
 │    │    └── settings.py              # Centralised thresholds and paths
 │    ├── api/
 │    │    ├── quality.py               # POST /quality-check route
 │    │    └── similarity.py            # POST /seal-scan/similarity route
 │    ├── services/
 │    │    ├── image_quality.py         # Quality checks engine
 │    │    ├── blur_classifier.py       # ML Blur Classifier service
 │    │    ├── preprocessing.py         # 512x512, Denoise, HSV, Canny edges
 │    │    ├── feature_extraction.py    # Pairwise similarity metrics
 │    │    ├── similarity_engine.py     # Orchestration of comparisons
 │    │    ├── reference_aggregation.py # best_match / mean / median
 │    │    └── tampering_classifier.py  # ML tampering risk prediction
 │    ├── models/
 │    │    └── schemas.py               # Pydantic JSON request/response models
 │    ├── utils/
 │    │    └── image_utils.py           # Base64 decode, data-URI sanitisation
 │    └── main.py                       # FastAPI application & startup warmup
 ├── model/
 │    ├── blur_classifier.pkl           # Pretrained ML blur classifier
 │    ├── blur_classifier_metadata.json # Blur model specification & metadata
 │    ├── tampering_classifier.pkl      # Pretrained ML tampering classifier
 │    └── generate_demo_model.py        # Demo tampering model generator
 ├── tests/
 │    ├── test_quality.py               # Quality unit tests
 │    ├── test_similarity.py            # Feature extraction & ML tests
 │    └── test_api.py                   # FastAPI integration tests
 ├── requirements.txt
 ├── workflow.md                        # Architecture & schema documentation
 └── README.md
```
