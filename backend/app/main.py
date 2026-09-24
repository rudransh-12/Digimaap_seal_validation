"""
SealScan Backend -- FastAPI Application Entry Point.

Startup:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations
import logging
import logging.config
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config.settings import LOG_LEVEL, LOG_FORMAT

# ---------------------------------------------------------------------------
# Logging configuration (set up BEFORE importing services)
# ---------------------------------------------------------------------------
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("sealscan.main")

# ---------------------------------------------------------------------------
# Import routers
# ---------------------------------------------------------------------------
from app.api.quality import router as quality_router
from app.api.similarity import router as similarity_router

# ---------------------------------------------------------------------------
# Pre-warm singletons at startup
# ---------------------------------------------------------------------------
def _warmup() -> None:
    """Load classifier and engine once so the first request is not slow."""
    logger.info("Starting up SealScan backend -- warming up services...")
    from app.api.similarity import _get_engine, _get_classifier
    _get_engine()
    _get_classifier()
    logger.info("Warmup complete.")

# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _warmup()
    yield
    logger.info("SealScan backend shutting down.")

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SealScan API",
    description=(
        "AI-Assisted Legal Metrology Seal Verification System.\n\n"
        "## Disclaimer\n"
        "All tampering risk assessments are **AI-assisted decision-support tools** only. "
        "They do NOT constitute a legal determination. "
        "The inspecting officer''s final judgment is authoritative.\n\n"
        "## Endpoints\n"
        "- `POST /quality-check` — Validate image quality only\n"
        "- `POST /seal-scan/similarity` — Full similarity + tampering assessment\n"
    ),
    version="1.0.0",
    contact={"name": "SealScan Team"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round(time.perf_counter() - start, 4)
    response.headers["X-Process-Time"] = str(elapsed)
    return response

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected server error occurred.",
            },
        },
    )

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(quality_router)
app.include_router(similarity_router)

# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {"service": "SealScan", "status": "ok", "docs": "/docs"}


@app.get("/health", tags=["Health"], summary="Health check")
async def health():
    return {"status": "healthy", "service": "SealScan"}


