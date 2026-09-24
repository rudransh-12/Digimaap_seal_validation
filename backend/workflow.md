# SealScan Backend — Architecture, Schema & Workflow

> **AI-Assisted Legal Metrology Seal Verification System**
> All outputs are decision-support only. The inspecting officer's determination is authoritative.

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Flutter Client (Mobile App)                        │
│                                                                             │
│   POST /quality-check              POST /seal-scan/similarity               │
│   ┌──────────────────┐             ┌─────────────────────────────────────┐  │
│   │   image (file)   │             │   current_image (file)              │  │
│   └──────────────────┘             │   reference_images[] (files)        │  │
│                                    └─────────────────────────────────────┘  │
└────────────────────┬───────────────────────────┬────────────────────────────┘
                     │  multipart/form-data       │  multipart/form-data
                     ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application  (:8000)                        │
│                                                                             │
│  ┌──────────────────┐             ┌──────────────────────────────────────┐  │
│  │  api/quality.py  │             │        api/similarity.py             │  │
│  │  POST            │             │        POST /seal-scan/similarity     │  │
│  │  /quality-check  │             └──────────────────────────────────────┘  │
│  └──────────────────┘                                                       │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          Services Layer                              │   │
│  │                                                                      │   │
│  │  ┌───────────────────┐   ┌─────────────────┐   ┌─────────────────┐  │   │
│  │  │  image_quality.py │   │ preprocessing.py│   │feature_extract..│  │   │
│  │  │  - Resolution     │   │ - Resize 512x512│   │ - Cosine Sim    │  │   │
│  │  │  - Sharpness      │   │ - Gaussian blur │   │ - ORB + RANSAC  │  │   │
│  │  │  - Brightness     │   │ - Grayscale     │   │ - SSIM          │  │   │
│  │  │  - Contrast       │   │ - HSV           │   │ - Edge Diff     │  │   │
│  │  │  - Noise          │   │ - Norm float    │   │ - Hist Diff     │  │   │
│  │  └───────────────────┘   │ - Canny edges   │   │ - Shape (Hu)    │  │   │
│  │                          └─────────────────┘   └─────────────────┘  │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────┐   ┌─────────────────────────────────┐  │   │
│  │  │ reference_aggregation.py│   │   tampering_classifier.py       │  │   │
│  │  │ - best_match (default)  │   │   - Load .pkl once at startup   │  │   │
│  │  │ - mean                  │   │   - Heuristic fallback if no .pk │  │   │
│  │  │ - median                │   │   - Outputs score + risk class  │  │   │
│  │  └─────────────────────────┘   └─────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  config/settings.py — All thresholds, paths, and parameters here   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 model/tampering_classifier.pkl
                 (loaded once at startup, never reloaded)
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `app/main.py` | FastAPI app, CORS, request timing middleware, lifespan startup warmup |
| `app/config/settings.py` | **Single source of truth** for all thresholds and parameters |
| `app/api/quality.py` | Thin route for `POST /quality-check` |
| `app/api/similarity.py` | Thin route for `POST /seal-scan/similarity` |
| `app/services/image_quality.py` | Evaluates image against quality thresholds; used by **both** endpoints |
| `app/services/preprocessing.py` | Resize, denoise, grayscale, HSV, edge map; produces `PreprocessedImage` |
| `app/services/feature_extraction.py` | Computes the 6 similarity metrics for one image pair |
| `app/services/similarity_engine.py` | Orchestrates preprocessing + extraction over all reference images |
| `app/services/reference_aggregation.py` | Aggregates per-reference results (best_match / mean / median) |
| `app/services/tampering_classifier.py` | Loads `.pkl`, falls back to heuristic, outputs score + risk class |
| `app/models/schemas.py` | All Pydantic request/response models |
| `app/utils/image_utils.py` | File validation (type, size, decode), colour conversions |
| `model/tampering_classifier.pkl` | Pre-trained sklearn RandomForest; loaded once at startup |
| `model/generate_demo_model.py` | One-time script to produce the demo `.pkl` from synthetic data |

---

## 2. Data Schemas

### 2A. Quality Metric (used in both endpoints)

```json
{
  "metric":    "sharpness",
  "value":     245.62,
  "threshold": 80.0,
  "passed":    true,
  "message":   null
}
```

| Field | Type | Description |
|---|---|---|
| `metric` | string | Metric name (sharpness / brightness / contrast / noise / resolution) |
| `value` | float | Measured value |
| `threshold` | float \| null | Acceptance threshold |
| `passed` | bool | True if metric is within acceptable range |
| `message` | string \| null | Human-readable explanation when failed |

---

### 2B. Resolution Detail

```json
{
  "width":  1920,
  "height": 1080,
  "passed": true
}
```

---

### 2C. All Metrics (quality-check success)

```json
{
  "resolution": { "width": 1920, "height": 1080, "passed": true },
  "sharpness":  { "metric": "sharpness",  "value": 245.62, "threshold": 80.0,  "passed": true },
  "brightness": { "metric": "brightness", "value": 128.4,  "threshold": 225.0, "passed": true },
  "contrast":   { "metric": "contrast",   "value": 57.2,   "threshold": 20.0,  "passed": true },
  "noise":      { "metric": "noise",      "value": 3.1,    "threshold": 15.0,  "passed": true }
}
```

---

### 2D. Per-Reference Comparison

```json
{
  "reference_id":        "reference_1",
  "cosine_similarity":   0.97,
  "orb_match_ratio":     0.81,
  "ssim_score":          0.94,
  "edge_difference":     0.04,
  "histogram_difference":0.06,
  "shape_difference":    0.02
}
```

| Metric | Range | Better When |
|---|---|---|
| `cosine_similarity` | [0, 1] | Higher |
| `orb_match_ratio` | [0, 1] | Higher |
| `ssim_score` | [0, 1] | Higher |
| `edge_difference` | [0, 1] | Lower |
| `histogram_difference` | [0, 1] | Lower |
| `shape_difference` | [0, 1] | Lower |

---

### 2E. Tampering Assessment

```json
{
  "tampering_score": 0.12,
  "risk_class": "LOW"
}
```

| Field | Type | Description |
|---|---|---|
| `tampering_score` | float [0,1] | Higher = higher tampering risk |
| `risk_class` | LOW \| MEDIUM \| HIGH | Risk classification |

> ⚠️ `tampering_score` is **not a calibrated probability** unless the deployed model has been explicitly calibrated with `CalibratedClassifierCV`.

---

### 2F. POST /quality-check — Success Response

```json
{
  "success": true,
  "quality_passed": true,
  "message": "Image quality is satisfactory.",
  "metrics": {
    "resolution": { "width": 1920, "height": 1080, "passed": true },
    "sharpness":  { "metric": "sharpness",  "value": 245.62, "threshold": 80.0,  "passed": true },
    "brightness": { "metric": "brightness", "value": 128.4,  "threshold": 225.0, "passed": true },
    "contrast":   { "metric": "contrast",   "value": 57.2,   "threshold": 20.0,  "passed": true },
    "noise":      { "metric": "noise",      "value": 3.1,    "threshold": 15.0,  "passed": true }
  }
}
```

---

### 2G. POST /quality-check — Failure Response

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
      "metric": "brightness",
      "value": 238.4,
      "threshold": 220.0,
      "passed": false,
      "message": "Image is overexposed. Reduce light source or adjust camera."
    }
  ]
}
```

---

### 2H. POST /seal-scan/similarity — Success Response

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
    },
    {
      "reference_id":         "reference_2",
      "cosine_similarity":    0.88,
      "orb_match_ratio":      0.70,
      "ssim_score":           0.87,
      "edge_difference":      0.09,
      "histogram_difference": 0.11,
      "shape_difference":     0.05
    }
  ]
}
```

---

### 2I. POST /seal-scan/similarity — Quality Failure Response

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

> No similarity metrics or tampering assessment are returned when quality fails.

---

### 2J. Error Response (any endpoint)

```json
{
  "success": false,
  "error": {
    "code": "INVALID_FILE_TYPE",
    "message": "Unsupported file type '.pdf'. Allowed types: .jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp"
  }
}
```

| Error Code | HTTP | Trigger |
|---|---|---|
| `INVALID_FILE_TYPE` | 422 | Unsupported file extension |
| `EMPTY_FILE` | 422 | Zero-byte upload |
| `FILE_TOO_LARGE` | 413 | File > 20 MB |
| `INVALID_IMAGE` | 422 | Cannot be decoded as image |
| `MISSING_REFERENCE_IMAGES` | 422 | No reference images provided |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

---

## 3. Workflows

### Workflow A — POST /quality-check

```
Flutter Client
      │
      │  POST /quality-check
      │  multipart/form-data
      │  ┌─────────────┐
      │  │  image file │
      │  └─────────────┘
      │
      ▼
┌─────────────────────────────┐
│  1. File Validation         │
│     - extension check       │
│     - size check (≤ 20 MB)  │
│     - empty file check      │
└────────────┬────────────────┘
             │ FAIL → 422 Error (INVALID_FILE_TYPE / EMPTY_FILE / FILE_TOO_LARGE)
             │
             ▼
┌─────────────────────────────┐
│  2. Image Decode            │
│     cv2.imdecode → BGR array│
└────────────┬────────────────┘
             │ FAIL → 422 Error (INVALID_IMAGE)
             │
             ▼
┌───────────────────────────────────────────────────────┐
│  3. Image Quality Service  (image_quality.py)         │
│                                                       │
│  ┌──────────────────┐   ┌───────────────────────┐    │
│  │  Resolution      │   │  width ≥ 320 px        │    │
│  │  (pixel dims)    │   │  height ≥ 320 px       │    │
│  └──────────────────┘   └───────────────────────┘    │
│                                                       │
│  ┌──────────────────┐   ┌───────────────────────┐    │
│  │  Sharpness       │   │  Laplacian variance    │    │
│  │                  │   │  ≥ 80.0               │    │
│  └──────────────────┘   └───────────────────────┘    │
│                                                       │
│  ┌──────────────────┐   ┌───────────────────────┐    │
│  │  Brightness      │   │  mean pixel            │    │
│  │                  │   │  30.0 ≤ val ≤ 225.0   │    │
│  └──────────────────┘   └───────────────────────┘    │
│                                                       │
│  ┌──────────────────┐   ┌───────────────────────┐    │
│  │  Contrast        │   │  std-dev pixels        │    │
│  │                  │   │  ≥ 20.0               │    │
│  └──────────────────┘   └───────────────────────┘    │
│                                                       │
│  ┌──────────────────┐   ┌───────────────────────┐    │
│  │  Noise           │   │  Gaussian residual     │    │
│  │                  │   │  std-dev ≤ 15.0        │    │
│  └──────────────────┘   └───────────────────────┘    │
│                                                       │
│  ALL failures collected before returning              │
└────────┬──────────────────────────────┬───────────────┘
         │ FAIL                         │ PASS
         ▼                              ▼
┌──────────────────────┐    ┌────────────────────────────────┐
│  200 Response        │    │  200 Response                  │
│  quality_passed=false│    │  quality_passed=true           │
│  failed_metrics=[...]│    │  metrics={ resolution,         │
│                      │    │    sharpness, brightness,      │
│  Flutter shows       │    │    contrast, noise }           │
│  specific error msg  │    │                                │
│  to officer          │    │  Flutter proceeds with         │
└──────────────────────┘    │  inspection                    │
                            └────────────────────────────────┘
```

---

### Workflow B — POST /seal-scan/similarity (Quality Fail Path)

```
Flutter Client
      │
      │  POST /seal-scan/similarity
      │  ┌─────────────────────┐
      │  │  current_image      │
      │  │  reference_images[] │
      │  └─────────────────────┘
      │
      ▼
  File Validation (current + all reference files)
      │ any FAIL → 422 Error
      │
      ▼
  Image Quality Check on current_image only
      │
      │ FAIL
      ▼
┌──────────────────────────────────┐
│  200 Response                    │
│  success=false                   │
│  quality_passed=false            │
│  failed_metrics=[...]            │
│                                  │
│  ← No similarity metrics         │
│  ← No tampering assessment       │
│  ← No reference images processed │
└──────────────────────────────────┘
```

---

### Workflow C — POST /seal-scan/similarity (Full Pipeline)

```
Flutter Client
      │
      │  POST /seal-scan/similarity
      │  ┌──────────────────────────────┐
      │  │  current_image               │
      │  │  reference_images[]          │
      │  │  (1 or more)                 │
      │  └──────────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│  1. File Validation      │
│     - all uploads        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────────────────────┐
│  2. Image Quality Check (current only)   │
│     (same service used by /quality-check)│
└────────────┬─────────────────────────────┘
             │ PASS
             ▼
┌──────────────────────────────────────────────────────────────┐
│  3. Preprocessing  (preprocessing.py → PreprocessedImage)    │
│                                                              │
│  Applied to CURRENT image once, then to each REFERENCE image │
│                                                              │
│  Step 1: Resize to 512×512 (INTER_AREA)                     │
│  Step 2: Gaussian blur (kernel=3) for noise reduction        │
│  Step 3: Convert to Grayscale                               │
│  Step 4: Convert to HSV (colour histogram)                  │
│  Step 5: Normalise grayscale to float [0,1] (cosine)        │
│  Step 6: Canny edge map (Otsu auto-threshold)               │
│                                                              │
│  Output: PreprocessedImage { bgr, gray, hsv, edges, gray_n } │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│  4. Feature Extraction (feature_extraction.py)               │
│                                                              │
│  For each reference image:                                   │
│                                                              │
│  current ──┬──▶  A. cosine_similarity                       │
│             │       vector dot product of gray_norm arrays   │
│             │                                                │
│             ├──▶  B. orb_match_ratio                        │
│             │       ORB keypoints → BFMatcher → Lowe ratio  │
│             │       → RANSAC homography → good/total         │
│             │                                                │
│             ├──▶  C. ssim_score                             │
│             │       skimage SSIM on grayscale pair           │
│             │                                                │
│             ├──▶  D. edge_difference                        │
│             │       mean |edges_cur - edges_ref| normalised  │
│             │                                                │
│             ├──▶  E. histogram_difference                   │
│             │       HSV H+S histogram Bhattacharyya dist     │
│             │                                                │
│             └──▶  F. shape_difference                       │
│                      log-Hu-moments L2 distance / 30         │
│                                                              │
│  Returns one metrics dict per reference                      │
└────────────┬─────────────────────────────────────────────────┘
             │
             │  reference_1 → { cos, orb, ssim, edge, hist, shape }
             │  reference_2 → { cos, orb, ssim, edge, hist, shape }
             │  reference_N → { cos, orb, ssim, edge, hist, shape }
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│  5. Reference Aggregation  (reference_aggregation.py)        │
│                                                              │
│  Strategy (config: REFERENCE_AGGREGATION_METHOD)             │
│                                                              │
│  ┌─────────────┐  Select reference with best SSIM score      │
│  │ best_match  │  (configurable via BEST_MATCH_METRIC)        │
│  │ (default)   │  Use its metrics as the aggregated vector    │
│  └─────────────┘                                             │
│                                                              │
│  ┌─────────────┐  Average each metric across all references  │
│  │    mean     │                                             │
│  └─────────────┘                                             │
│                                                              │
│  ┌─────────────┐  Median each metric across all references   │
│  │   median    │                                             │
│  └─────────────┘                                             │
│                                                              │
│  Output: aggregated_metrics { 6 scalar values }              │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│  6. Tampering Classifier  (tampering_classifier.py)          │
│                                                              │
│  Input feature vector (order from settings.py):             │
│  [ cosine_similarity, orb_match_ratio, ssim_score,          │
│    edge_difference, histogram_difference, shape_difference ] │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  model/tampering_classifier.pkl  (loaded at startup) │   │
│  │  sklearn RandomForestClassifier                       │   │
│  │  predict_proba → score + class index                 │   │
│  └────────────────────────────────────┬─────────────────┘   │
│                                       │ if .pkl missing      │
│  ┌────────────────────────────────────▼─────────────────┐   │
│  │  Heuristic Fallback                                   │   │
│  │  Weighted sum of metrics → normalised risk score      │   │
│  │  score ≤ 0.35  → LOW                                 │   │
│  │  score ≤ 0.70  → MEDIUM                              │   │
│  │  score > 0.70  → HIGH                                │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  Output: tampering_score (float [0,1])                       │
│          risk_class (LOW | MEDIUM | HIGH)                    │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│  7. JSON Response                                            │
│                                                              │
│  {                                                           │
│    "success": true,                                          │
│    "quality_passed": true,                                   │
│    "message": "...Tampering risk assessment: LOW...",        │
│    "tampering_assessment": {                                 │
│       "tampering_score": 0.12,                              │
│       "risk_class": "LOW"                                   │
│    },                                                        │
│    "aggregated_metrics": { ...6 metrics... },               │
│    "reference_comparisons": [ ...per-reference details... ] │
│  }                                                           │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Quality Thresholds Reference

All values are defined in `app/config/settings.py` under `QUALITY_THRESHOLDS`.

| Metric | Method | Threshold | Pass Condition |
|---|---|---|---|
| Resolution | Pixel dimensions | min 320×320 | width ≥ 320 AND height ≥ 320 |
| Sharpness | Laplacian variance | ≥ 80.0 | Higher = sharper |
| Brightness | Mean greyscale pixel | 30.0 – 225.0 | Under/over-exposure both fail |
| Contrast | Std-dev of greyscale | ≥ 20.0 | Low std = washed-out image |
| Noise | Gaussian blur residual std | ≤ 15.0 | High residual = noisy sensor |

---

## 5. Similarity Metrics Reference

All six metrics are computed by `app/services/feature_extraction.py`.

| Metric | Algorithm | Better When | Notes |
|---|---|---|---|
| `cosine_similarity` | Dot product of flattened normalised greyscale arrays | Higher | Fast global similarity |
| `orb_match_ratio` | ORB + BFMatcher + Lowe ratio + RANSAC homography | Higher | Robust to rotation & scale |
| `ssim_score` | Structural Similarity Index (luminance, contrast, structure) | Higher | Perceptual similarity |
| `edge_difference` | Mean \|Canny(cur) − Canny(ref)\| normalised to [0,1] | Lower | Edge/outline comparison |
| `histogram_difference` | Bhattacharyya on HSV H+S histograms | Lower | Colour distribution, illumination-robust |
| `shape_difference` | L2 of log-Hu-moment vectors, normalised by 30 | Lower | Global shape/contour comparison |

---

## 6. Classifier Risk Classes

| Class | Typical Score Range | Interpretation |
|---|---|---|
| `LOW` | 0.00 – 0.35 | Metrics consistent with an intact, unmodified seal |
| `MEDIUM` | 0.36 – 0.70 | Moderate differences detected; officer review recommended |
| `HIGH` | 0.71 – 1.00 | Significant differences detected; physical inspection required |

> **Legal Disclaimer:** These classifications are AI-assisted risk indicators, not legal determinations. The inspecting officer's judgment is always authoritative.

---

## 7. Separation of Responsibilities

```
API Layer  (api/quality.py, api/similarity.py)
     │  Thin routes: validate inputs, call services, return JSON
     ▼
Validation  (utils/image_utils.py)
     │  File type, size, decode checks
     ▼
Image Quality Service  (services/image_quality.py)
     │  Used by BOTH endpoints; never skipped
     ▼
Preprocessing  (services/preprocessing.py)
     │  Only reached after quality PASS in similarity endpoint
     ▼
Feature Extraction  (services/feature_extraction.py)
     │  6 metrics per image pair; pure functions
     ▼
Similarity Engine  (services/similarity_engine.py)
     │  Loops over all reference images; sequential processing
     ▼
Reference Aggregation  (services/reference_aggregation.py)
     │  Isolated strategy — change without touching extraction
     ▼
Tampering Classifier  (services/tampering_classifier.py)
     │  Loaded once at startup; never retrained at request time
     ▼
JSON Response
```
