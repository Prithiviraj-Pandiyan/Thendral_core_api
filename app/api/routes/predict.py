from fastapi import APIRouter
from pydantic import BaseModel
from ML.inference import SpamInferenceService

router = APIRouter()
inference_service = SpamInferenceService()

class Predictionrequest(BaseModel):
    text : str

@router.post("/")
def predict_endpoint(request:Predictionrequest):
    result = inference_service.predict(request.text)
    return result
