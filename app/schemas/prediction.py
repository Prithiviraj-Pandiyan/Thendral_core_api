from typing import Literal

from pydantic import BaseModel

from ML.core.tasks import TaskKey
from app.schemas.dataset import ModelKey


class PredictionRequest(BaseModel):
    text: str
    task_key: TaskKey = TaskKey.SPAM_DETECTION
    model_key: ModelKey = ModelKey.logistic_regression


class SpamDetectionPrediction(BaseModel):
    input_text: str
    detected_html: bool | None = None
    cleaned_text: str
    is_spam: bool
    spam_label: Literal["spam", "ham"]
    spam_confidence: float
    ham_intent: str | None = None
    ham_intent_confidence: float | None = None


class HamIntentPrediction(BaseModel):
    input_text: str
    detected_html: bool | None = None
    cleaned_text: str
    ham_intent: str
    ham_intent_confidence: float


class SpamDetectionPredictResponse(BaseModel):
    task_key: Literal["spam_detection"]
    model_key: ModelKey
    prediction: SpamDetectionPrediction


class HamIntentPredictResponse(BaseModel):
    task_key: Literal["ham_intent"]
    model_key: ModelKey
    prediction: HamIntentPrediction
