from fastapi import APIRouter
from pydantic import BaseModel
from ML.inference import SpamInferenceService

router = APIRouter()
inference_service = SpamInferenceService()


class PredictionRequest(BaseModel):
    text: str


@router.post("/")
def predict_endpoint(request: PredictionRequest):
    result = inference_service.predict(request.text)
    return result
