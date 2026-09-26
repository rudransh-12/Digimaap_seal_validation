"""
SealScan Backend -- Centralised Configuration
All tunable parameters live here. Change thresholds / paths in this file only.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]   # project root (backend/)
MODEL_DIR = BASE_DIR / "model"
CLASSIFIER_PATH = MODEL_DIR / "tampering_classifier.pkl"
BLUR_CLASSIFIER_PATH = MODEL_DIR / "blur_classifier.pkl"
BLUR_CLASSIFIER_METADATA_PATH = MODEL_DIR / "blur_classifier_metadata.json"

# Feature order expected by the blur classifier Random Forest
BLUR_CLASSIFIER_FEATURE_ORDER: list[str] = [
    "laplacian_variance",
    "edge_strength",
    "noise",
    "brenner_sharpness",
    "canny_edge_density",
    "fft_high_frequency_ratio",
    "blur_effect",
]

# ---------------------------------------------------------------------------
# API / File-upload limits
# ---------------------------------------------------------------------------
MAX_IMAGE_SIZE_BYTES: int = 20 * 1024 * 1024          # 20 MB
ALLOWED_MIME_TYPES: list[str] = [
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/webp",
]
ALLOWED_EXTENSIONS: list[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

# ---------------------------------------------------------------------------
# Image Quality Thresholds
# ---------------------------------------------------------------------------
QUALITY_THRESHOLDS: dict = {
    # Resolution
    "min_width": 300,
    "min_height": 300,
    # Sharpness -- Laplacian variance (higher = sharper)
    "min_sharpness": 80.0,
    # Brenner sharpness -- mean squared difference with step 2 (higher = sharper)
    "min_brenner_sharpness": 100.0,
    # Canny Edge Density -- fraction of edge pixels in [0.0, 1.0]
    "min_canny_edge_density": 0.008,
    # Blur Classifier ML model -- min probability of being NOT_BLURRY in [0.0, 1.0]
    "min_blur_classifier_score": 0.50,
    # Brightness -- mean greyscale pixel value
    "min_brightness": 30.0,
    "max_brightness": 225.0,
    # Contrast -- std-dev of greyscale values
    "min_contrast": 25.0,
    # Noise estimate
    "max_noise": 15.0,
}

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
PREPROCESSING: dict = {
    "target_size": (512, 512),      # (width, height)
    "gaussian_blur_kernel": 3,      # odd integer; 0 to skip
}

# ---------------------------------------------------------------------------
# SIFT Feature Matching
# ---------------------------------------------------------------------------
SIFT_CONFIG: dict = {
    "n_features": 0,                # 0 = unconstrained detection
    "n_octave_layers": 3,
    "contrast_threshold": 0.04,
    "edge_threshold": 10.0,
    "sigma": 1.6,
    "lowe_ratio": 0.75,
    "ransac_reproj_threshold": 5.0,
    "min_good_matches": 4,
}

# ---------------------------------------------------------------------------
# Reference Image Aggregation
# ---------------------------------------------------------------------------
# Options: "best_match" | "mean" | "median"
REFERENCE_AGGREGATION_METHOD: str = "best_match"

# For "best_match": which metric to rank by?
BEST_MATCH_METRIC: str = "sift_match_ratio"

# ---------------------------------------------------------------------------
# Tampering Classifier
# ---------------------------------------------------------------------------
CLASSIFIER_FEATURE_ORDER: list[str] = [
    "cosine_similarity",
    "sift_match_ratio",
    "ssim_score",
    "edge_difference",
    "histogram_difference",
    "shape_difference",
]

RISK_CLASSES: list[str] = ["LOW", "MEDIUM", "HIGH"]

# Heuristic thresholds (fallback when no .pkl exists)
HEURISTIC_THRESHOLDS: dict = {
    "LOW": 0.35,
    "MEDIUM": 0.70,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
