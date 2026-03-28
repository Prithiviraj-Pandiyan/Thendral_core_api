from fastapi import APIRouter
from fastapi import HTTPException
from ML.core.tasks import TaskKey
from ML.inference import get_inference_service
from app.schemas.prediction import (
    HamIntentPredictResponse,
    PredictionRequest,
    SpamDetectionPredictResponse,
)

router = APIRouter()


@router.post("/", response_model=SpamDetectionPredictResponse | HamIntentPredictResponse)
def predict_endpoint(request: PredictionRequest):
    task_key = request.task_key.value
    model_key = request.model_key.value
    service = get_inference_service(task_key=task_key, model_key=model_key)
    result = service.predict(request.text)
    if request.task_key == TaskKey.SPAM_DETECTION:
        return SpamDetectionPredictResponse(
            task_key="spam_detection",
            model_key=request.model_key,
            prediction=result,
        )
    if request.task_key == TaskKey.HAM_INTENT:
        return HamIntentPredictResponse(
            task_key="ham_intent",
            model_key=request.model_key,
            prediction=result,
        )
    raise HTTPException(status_code=400, detail=f"Unsupported task_key: {task_key}")
