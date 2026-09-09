from enum import Enum
from pydantic import BaseModel, create_model
import config

class SentimentLabel(str, Enum):
    POS = "POS"
    NEG = "NEG"
    ABSENT = "ABSENT"


class ReviewClassification(BaseModel):
    DRIVER_OPS : SentimentLabel
    APP_AND_MAPS : SentimentLabel
    PRICING_AND_BILLING : SentimentLabel
    FULFILLMENT_FOOD : SentimentLabel