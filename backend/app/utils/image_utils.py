"""
SealScan -- Image utility helpers.
"""
from __future__ import annotations
import io
import logging
import os

import cv2
import numpy as np
from fastapi import UploadFile, HTTPException, status

from app.config.settings import (
    MAX_IMAGE_SIZE_BYTES,
    ALLOWED_EXTENSIONS,
)

logger = logging.getLogger("sealscan.image_utils")


def validate_upload(upload: UploadFile) -> bytes:
    """
    Validate an uploaded file:
    - extension check
    - size check
    Returns raw bytes for further processing.
    Raises HTTPException on validation failure.
    """
    filename = upload.filename or ""
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "success": False,
                "error": {
                    "code": "INVALID_FILE_TYPE",
                    "message": (
                        f"Unsupported file type '{ext}'. "
                        f"Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
                    ),
                },
            },
        )

    data = upload.file.read()

    if len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "success": False,
                "error": {
                    "code": "EMPTY_FILE",
                    "message": "The uploaded file is empty.",
                },
            },
        )

    if len(data) > MAX_IMAGE_SIZE_BYTES:
        mb = MAX_IMAGE_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "success": False,
                "error": {
                    "code": "FILE_TOO_LARGE",
                    "message": f"File exceeds maximum allowed size of {mb} MB.",
                },
            },
        )

    return data


def bytes_to_bgr(data: bytes, label: str = "image") -> np.ndarray:
    """
    Decode raw bytes into a BGR NumPy array (OpenCV native format).
    Raises HTTPException if decoding fails.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "success": False,
                "error": {
                    "code": "INVALID_IMAGE",
                    "message": f"The {label} could not be decoded as a valid image.",
                },
            },
        )
    return img


def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def to_hsv(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
