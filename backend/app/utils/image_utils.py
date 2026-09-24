"""
SealScan -- Image utility helpers.
"""
from __future__ import annotations
import base64
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


def base64_to_bgr(b64_string: str, label: str = "image") -> np.ndarray:
    """
    Decode a base64-encoded image string into a BGR NumPy array.

    Accepts both:
      - plain base64:          "/9j/4AAQSkZJRgAB..."
      - data-URI prefix:       "data:image/jpeg;base64,/9j/4AAQ..."

    Raises HTTPException on empty input, bad base64, or undecodable image.
    """
    if not b64_string or not b64_string.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "success": False,
                "error": {
                    "code": "EMPTY_IMAGE",
                    "message": f"The {label} base64 string is empty.",
                },
            },
        )

    # Strip data-URI prefix if present (e.g. "data:image/jpeg;base64,")
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    try:
        raw = base64.b64decode(b64_string)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "success": False,
                "error": {
                    "code": "INVALID_BASE64",
                    "message": f"The {label} could not be decoded from base64.",
                },
            },
        )

    if len(raw) > MAX_IMAGE_SIZE_BYTES:
        mb = MAX_IMAGE_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "success": False,
                "error": {
                    "code": "FILE_TOO_LARGE",
                    "message": f"Decoded image exceeds maximum allowed size of {mb} MB.",
                },
            },
        )

    return bytes_to_bgr(raw, label=label)
