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
│   │ { image: b64 }   │             │ { current_image: b64,               │  │
│   └──────────────────┘             │   reference_images: [b64, ...] }    │  │
│                                    └─────────────────────────────────────┘  │
└────────────────────┬───────────────────────────┬────────────────────────────┘
                     │  Content-Type: application/json
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
│  │  │  - Laplacian Var  │   │ - Gaussian blur │   │ - SIFT + RANSAC │  │   │
│  │  │  - Brenner Focus  │   │ - Grayscale     │   │ - SSIM          │  │   │
│  │  │  - Canny Density  │   │ - HSV           │   │ - Edge Diff     │  │   │
│  │  │  - AI Blur Clf    │   │ - Norm float    │   │ - Hist Diff     │  │   │
│  │  │  - Brightness     │   │ - Canny edges   │   │ - Shape (Hu)    │  │   │
│  │  │  - Contrast       │   └─────────────────┘   └─────────────────┘  │   │
│  │  │  - Noise          │                                               │   │
│  │  └─────────┬─────────┘   ┌─────────────────┐   ┌─────────────────┐  │   │
│  │            │             │ reference_      │   │   tampering_    │  │   │
│  │            ▼             │ aggregation.py  │   │   classifier.py │  │   │
│  │  ┌───────────────────┐   │ - best_match    │   │   - Load .pkl   │  │   │
│  │  │ blur_classifier.py│   │ - mean          │   │   - Heuristic   │  │   │
│  │  │ (RandomForest ML) │   │ - median        │   │   - Risk score  │  │   │
│  │  └───────────────────┘   └─────────────────┘   └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  config/settings.py — All thresholds, paths, and parameters here   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
model/blur_classifier.pkl             model/tampering_classifier.pkl
(ML Blur gate; loaded at startup)     (Tampering classifier; loaded at startup)
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `app/main.py` | FastAPI app, CORS, request timing middleware, lifespan startup warmup |
| `app/config/settings.py` | **Single source of truth** for all thresholds, paths, and hyperparameters |
| `app/api/quality.py` | Thin route for `POST /quality-check` (accepts Base64 JSON) |
| `app/api/similarity.py` | Thin route for `POST /seal-scan/similarity` (accepts Base64 JSON) |
| `app/services/image_quality.py` | Evaluates image against quality thresholds & ML blur model; used by **both** endpoints |
| `app/services/blur_classifier.py` | 7-feature extraction & pretrained Random Forest inference for blur assessment |
| `app/services/preprocessing.py` | Resize, denoise, grayscale, HSV, edge map; produces `PreprocessedImage` |
| `app/services/feature_extraction.py` | Computes the 6 similarity metrics for one image pair |
| `app/services/similarity_engine.py` | Orchestrates preprocessing + extraction over all reference images |
| `app/services/reference_aggregation.py` | Aggregates per-reference results (best_match / mean / median) |
| `app/services/tampering_classifier.py` | Loads `.pkl`, falls back to heuristic, outputs score + risk class |
| `app/models/schemas.py` | All Pydantic request/response models |
| `app/utils/image_utils.py` | Base64 decode, data-URI sanitisation, image validation, colour conversions |
| `model/blur_classifier.pkl` | Pretrained RandomForestClassifier for defocus / motion blur detection |
| `model/blur_classifier_metadata.json` | Model metadata, feature definitions, and performance specs |
| `model/tampering_classifier.pkl` | Pretrained RandomForestClassifier for seal tampering risk prediction |
| `model/generate_demo_model.py` | One-time script to produce demo tampering `.pkl` from synthetic data |

---

## 2. Data Schemas

### 2A. Request Payloads (Base64 JSON)

#### POST /quality-check
```json
{
  "image": "<base64 string or data:image/...;base64,...>"
}
```

#### POST /seal-scan/similarity
```json
{
  "current_image": "<base64 string>",
  "reference_images": [
    "<base64 string 1>",
    "<base64 string 2>"
  ]
}
```

---

### 2B. Quality Metric Detail

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
| `metric` | string | Metric identifier (`sharpness`, `brenner_sharpness`, `canny_edge_density`, `blur_classifier`, `brightness`, `contrast`, `noise`, `resolution`) |
| `value` | float | Measured value |
| `threshold` | float \| null | Threshold value configured in settings |
| `passed` | bool | True if metric satisfies threshold |
| `message` | string \| null | Human-readable explanation / retake prompt when failed |

---

### 2C. Resolution Detail

```json
{
  "width":  1920,
  "height": 1080,
  "passed": true
}
```

---

### 2D. All Metrics Object (`AllMetrics`)

```json
{
  "resolution":        { "width": 1920, "height": 1080, "passed": true },
  "sharpness":         { "metric": "sharpness",          "value": 245.62, "threshold": 80.0,   "passed": true, "message": null },
  "brenner_sharpness": { "metric": "brenner_sharpness",  "value": 412.50, "threshold": 100.0,  "passed": true, "message": null },
  "canny_edge_density":{ "metric": "canny_edge_density", "value": 0.0452, "threshold": 0.008,  "passed": true, "message": null },
  "blur_classifier":   { "metric": "blur_classifier",   "value": 0.9820, "threshold": 0.50,   "passed": true, "message": null },
  "brightness":        { "metric": "brightness",        "value": 128.40, "threshold": 225.0,  "passed": true, "message": null },
  "contrast":          { "metric": "contrast",          "value": 57.20,  "threshold": 25.0,   "passed": true, "message": null },
  "noise":             { "metric": "noise",             "value": 3.10,   "threshold": 15.0,   "passed": true, "message": null }
}
```

---

### 2E. Per-Reference Comparison

```json
{
  "reference_id":        "reference_1",
  "cosine_similarity":   0.97,
  "sift_match_ratio":    0.81,
  "ssim_score":          0.94,
  "edge_difference":     0.04,
  "histogram_difference":0.06,
  "shape_difference":    0.02
}
```

| Metric | Range | Better When | Description |
|---|---|---|---|
| `cosine_similarity` | [0, 1] | Higher | Normalised greyscale dot product similarity |
| `sift_match_ratio` | [0, 1] | Higher | Geometric RANSAC-filtered SIFT keypoint match ratio |
| `ssim_score` | [0, 1] | Higher | Structural Similarity Index (luminance, contrast, structure) |
| `edge_difference` | [0, 1] | Lower | Mean absolute difference of Canny edge maps |
| `histogram_difference` | [0, 1] | Lower | Bhattacharyya distance of HSV colour histograms |
| `shape_difference` | [0, 1] | Lower | Log-Hu-moments distance |

---

### 2F. Tampering Assessment

```json
{
  "tampering_score": 0.12,
  "risk_class": "LOW"
}
```

| Field | Type | Description |
|---|---|---|
| `tampering_score` | float [0, 1] | Higher = higher tampering risk |
| `risk_class` | LOW \| MEDIUM \| HIGH | Risk categorization for decision support |

---

### 2G. POST /quality-check — Responses

#### Success Response (200 OK)
```json
{
  "success": true,
  "quality_passed": true,
  "message": "Image quality is satisfactory.",
  "metrics": {
    "resolution":        { "width": 1920, "height": 1080, "passed": true },
    "sharpness":         { "metric": "sharpness",         "value": 245.62, "threshold": 80.0,   "passed": true, "message": null },
    "brenner_sharpness": { "metric": "brenner_sharpness", "value": 412.50, "threshold": 100.0,  "passed": true, "message": null },
    "canny_edge_density":{ "metric": "canny_edge_density","value": 0.0452, "threshold": 0.008,  "passed": true, "message": null },
    "blur_classifier":   { "metric": "blur_classifier",  "value": 0.9820, "threshold": 0.50,   "passed": true, "message": null },
    "brightness":        { "metric": "brightness",       "value": 128.40, "threshold": 225.0,  "passed": true, "message": null },
    "contrast":          { "metric": "contrast",         "value": 57.20,  "threshold": 25.0,   "passed": true, "message": null },
    "noise":             { "metric": "noise",            "value": 3.10,   "threshold": 15.0,   "passed": true, "message": null }
  }
}
```

#### Quality Failure Response (200 OK)
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

### 2H. POST /seal-scan/similarity — Responses

#### Success Response (200 OK)
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
    "sift_match_ratio":     0.81,
    "ssim_score":           0.94,
    "edge_difference":      0.04,
    "histogram_difference": 0.06,
    "shape_difference":     0.02
  },
  "reference_comparisons": [
    {
      "reference_id":         "reference_1",
      "cosine_similarity":    0.97,
      "sift_match_ratio":     0.81,
      "ssim_score":           0.94,
      "edge_difference":      0.04,
      "histogram_difference": 0.06,
      "shape_difference":     0.02
    }
  ]
}
```

#### Quality Fail Response (200 OK)
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

### 2I. Error Envelope (4xx / 5xx)

```json
{
  "success": false,
  "error": {
    "code": "INVALID_IMAGE",
    "message": "The provided base64 data could not be decoded as a valid image."
  }
}
```

---

## 3. Workflows

### Workflow A — POST /quality-check

```
Flutter Client (JSON Payload)
      │
      │  POST /quality-check  { "image": "<base64>" }
      │
      ▼
┌────────────────────────────────────────────────────────┐
│  1. Base64 Decode & Validation                         │
│     - strip data-URI prefix if present                 │
│     - decode base64 bytes                              │
│     - size check (≤ 20 MB) & cv2.imdecode BGR check    │
└───────────┬────────────────────────────────────────────┘
            │ FAIL → 422 Error (EMPTY_FILE / FILE_TOO_LARGE / INVALID_IMAGE)
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  2. Image Quality Service  (image_quality.py)          │
│                                                        │
│  • Resolution: width ≥ 320 px, height ≥ 320 px         │
│  • Sharpness: Laplacian variance ≥ 80.0                │
│  • Brenner Sharpness: horizontal 2nd diff ≥ 100.0      │
│  • Canny Edge Density: edge ratio ≥ 0.008              │
│  • AI Blur Classifier: P(NOT_BLURRY) ≥ 0.50            │
│  • Brightness: mean pixel in [30.0, 225.0]             │
│  • Contrast: std-dev pixel in ≥ 25.0                   │
│  • Noise: Gaussian residual std ≤ 15.0                 │
│                                                        │
│  All 8 checks evaluated & all failures collected       │
└───────────┬────────────────────────────┬───────────────┘
            │ FAIL                       │ PASS
            ▼                            ▼
┌───────────────────────┐    ┌───────────────────────────┐
│ 200 Response          │    │ 200 Response              │
│ quality_passed=false  │    │ quality_passed=true       │
│ failed_metrics=[...]  │    │ metrics={ AllMetrics }    │
│                       │    │                           │
│ Flutter prompts user  │    │ Flutter proceeds with     │
│ to retake image       │    │ seal inspection           │
└───────────────────────┘    └───────────────────────────┘
```

---

### Workflow B — POST /seal-scan/similarity

```
Flutter Client (JSON Payload)
      │
      │  POST /seal-scan/similarity
      │  { "current_image": "<b64>", "reference_images": ["<b64>", ...] }
      │
      ▼
┌────────────────────────────────────────────────────────┐
│  1. Decode & Validate current_image + reference_images │
└───────────┬────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  2. Image Quality Check (current_image only)           │
└───────────┬────────────────────────────────────────────┘
            │
            ├─► FAIL ──► Return 200 { quality_passed: false, failed_metrics: [...] }
            │            (Similarity pipeline skipped)
            ▼ PASS
┌────────────────────────────────────────────────────────┐
│  3. Preprocessing (512×512, Denoise, HSV, Canny edges) │
└───────────┬────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  4. Pairwise Feature Extraction                        │
│     (Cosine, SIFT+RANSAC, SSIM, EdgeDiff, HistDiff, Hu) │
└───────────┬────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  5. Reference Aggregation                              │
│     (best_match / mean / median)                       │
└───────────┬────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  6. Tampering Classifier (Random Forest inference)     │
│     Output: tampering_score [0,1], risk_class          │
└───────────┬────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  7. JSON Response (Success + Tampering Assessment)     │
└────────────────────────────────────────────────────────┘
```

---

## 4. Quality Thresholds Reference

All values are configured in `app/config/settings.py` under `QUALITY_THRESHOLDS`:

| Metric | Method | Threshold | Pass Condition |
|---|---|---|---|
| Resolution | Pixel dimensions | min 320×320 | width ≥ 320 AND height ≥ 320 |
| Sharpness | Laplacian variance | ≥ 80.0 | Higher = sharper |
| Brenner Sharpness | 2nd-difference mean squared | ≥ 100.0 | Higher = sharper focus |
| Canny Edge Density | Ratio of edge pixels | ≥ 0.008 | Sufficient structural details |
| Blur Classifier | AI RandomForest model | ≥ 0.50 | Probability of NOT_BLURRY |
| Brightness | Mean greyscale pixel | 30.0 – 225.0 | Reject underexposed / overexposed |
| Contrast | Std-dev of greyscale pixels | ≥ 25.0 | Low std = washed-out / low dynamic range |
| Noise | Gaussian blur residual std | ≤ 15.0 | High residual = noisy sensor |

---

## 5. Blur Classifier Details

* **Model File:** `model/blur_classifier.pkl`
* **Metadata File:** `model/blur_classifier_metadata.json`
* **Model Type:** Scikit-learn `RandomForestClassifier` (150 trees, max depth 6)
* **Target Classes:** `0: BLURRY`, `1: NOT_BLURRY`
* **Extracted Features (7 in order):**
  1. `laplacian_variance`
  2. `edge_strength` (Sobel magnitude)
  3. `noise` (Gaussian residual)
  4. `brenner_sharpness` (horizontal second difference)
  5. `canny_edge_density` (Canny 100, 200)
  6. `fft_high_frequency_ratio` (2D FFT spectral energy)
  7. `blur_effect` (`skimage.measure.blur_effect`)

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
     │  Base64 decoding, size checks, image decode checks
     ▼
Image Quality Service  (services/image_quality.py)
     │  Used by BOTH endpoints; never skipped
     ▼
Blur Classifier  (services/blur_classifier.py)
     │  AI blur verification gate loaded once at startup
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
