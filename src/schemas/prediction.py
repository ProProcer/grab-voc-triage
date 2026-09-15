from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SentimentLabel(str, Enum):
    NEG = "NEG"
    ABSENT = "ABSENT"


class ReviewClassification(BaseModel):
    DRIVER_OPS: SentimentLabel
    APP_AND_MAPS: SentimentLabel
    PRICING_AND_BILLING: SentimentLabel


class CategoryScore(BaseModel):
    label: SentimentLabel
    probability: float = Field(..., ge=0.0, le=1.0, description="Confidence probability of NEG complaint")


class TriageRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Customer review or feedback text in Indonesian")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Classification threshold for NEG")


class TriageResponse(BaseModel):
    content: str
    categories: Dict[str, CategoryScore]
    active_complaints: List[str] = Field(
        default_factory=list,
        description="List of categories flagged as NEG (complaints requiring triage)",
    )


class BatchTriageRequest(BaseModel):
    contents: List[str] = Field(..., min_length=1, description="List of reviews to triage")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Classification threshold for NEG")


class BatchTriageResponse(BaseModel):
    results: List[TriageResponse]
    total_processed: int


class HealthResponse(BaseModel):
    status: str
    model_name: str
    device: str
    categories: List[str]