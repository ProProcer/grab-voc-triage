import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.inference.pipeline import TriagePipeline
from src.schemas.prediction import (
    BatchTriageRequest,
    BatchTriageResponse,
    HealthResponse,
    TriageRequest,
    TriageResponse,
)

# Global pipeline holder
pipeline: Optional[TriagePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and loads the model into memory upon server startup."""
    global pipeline
    checkpoint_path = os.getenv("CHECKPOINT_PATH", "checkpoints/indobert-base-run1/best_model.pt")
    pretrained_weights = os.getenv("PRETRAINED_MODEL", "pretrained_weights/indobert-base-p1")
    device = os.getenv("DEVICE", None)

    print(f"Loading IndoBERT triage pipeline from: {checkpoint_path}...")
    pipeline = TriagePipeline(
        checkpoint_path=checkpoint_path,
        pretrained_model=pretrained_weights,
        device=device,
    )
    print(f"Pipeline initialized successfully on device: {pipeline.device}")
    yield
    print("Shutting down IndoBERT triage service...")


app = FastAPI(
    title="Grab Voice-of-Customer (VOC) Triage Service",
    description=(
        "Production-grade microservice for multi-label negative review triage "
        "using fine-tuned IndoBERT (DRIVER_OPS, APP_AND_MAPS, PRICING_AND_BILLING)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Grab VOC Triage Service",
        "model": "IndoBERT-base (fine-tuned)",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health",
        "endpoints": {
            "single_triage": "POST /triage",
            "batch_triage": "POST /triage/batch",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not yet initialized.",
        )
    return HealthResponse(
        status="healthy",
        model_name="indobert-base-p1-grab-triage",
        device=str(pipeline.device),
        categories=pipeline.categories,
    )


@app.post("/triage", response_model=TriageResponse, tags=["Inference"])
def triage_review(request: TriageRequest):
    """
    Classify a single Indonesian customer review into triage categories.
    Returns per-category probabilities and active complaints (NEG).
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not initialized.",
        )
    return pipeline.predict_one(text=request.content, threshold=request.threshold or 0.5)


@app.post("/triage/batch", response_model=BatchTriageResponse, tags=["Inference"])
def triage_batch(request: BatchTriageRequest):
    """
    High-throughput batched triage for multiple reviews.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not initialized.",
        )
    results = pipeline.predict_batch(
        texts=request.contents,
        threshold=request.threshold or 0.5,
    )
    return BatchTriageResponse(results=results, total_processed=len(results))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("app:app", host=host, port=port, reload=False)
