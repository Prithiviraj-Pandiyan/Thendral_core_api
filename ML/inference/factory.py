from ML.core.tasks import TaskKey
from ML.inference.engine import InferenceEngine
from ML.inference.services.email_service import EmailInferenceService

_SERVICE_CACHE: dict[tuple[str, str], InferenceEngine] = {}


def get_inference_service(task_key: str, model_key: str) -> InferenceEngine:
    """Return cached inference service for task/model."""
    cache_key = (task_key, model_key)
    if cache_key in _SERVICE_CACHE:
        return _SERVICE_CACHE[cache_key]

    if task_key in {TaskKey.SPAM_DETECTION.value, TaskKey.HAM_INTENT.value}:
        service = EmailInferenceService(task_key=task_key, model_key=model_key)
        _SERVICE_CACHE[cache_key] = service
        return service

    raise ValueError(f"Unsupported task_key: {task_key}")
