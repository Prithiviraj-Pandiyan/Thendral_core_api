from fastapi import APIRouter, Depends, File, Form, UploadFile
from app.services.auth import require_admin_key
from ML.core.tasks import TaskKey

from app.schemas.dataset import (
    DatasetTrainingRequest,
    HuggingFaceIngestRequest,
    LabelColumnKey,
)
from app.services.dataset_service import (
    ingest_huggingface_dataset_service,
    list_datasets_service,
    train_dataset_service,
    upload_dataset_service,
)

router = APIRouter()


@router.post("/upload", dependencies=[Depends(require_admin_key)])
async def upload_file(
    task_key: TaskKey = Form(TaskKey.SPAM_DETECTION),
    text_column: str = Form("text"),
    label_column: LabelColumnKey = Form(LabelColumnKey.label),
    file: UploadFile = File(...),
):
    return await upload_dataset_service(
        file=file,
        task_key=task_key.value,
        text_column=text_column,
        label_column=label_column.value,
    )


@router.post("/huggingface", dependencies=[Depends(require_admin_key)])
def upload_huggingface_dataset(request: HuggingFaceIngestRequest):
    return ingest_huggingface_dataset_service(
        task_key=request.task_key,
        dataset_name=request.dataset_name,
        split=request.split,
        text_column=request.text_column,
        label_column=request.label_column,
        config_name=request.config_name,
        revision=request.revision,
    )


@router.get("/", dependencies=[Depends(require_admin_key)])
def list_datasets():
    return list_datasets_service()


@router.post("/train", dependencies=[Depends(require_admin_key)])
def train_from_dataset(request: DatasetTrainingRequest):
    return train_dataset_service(
        dataset_id = request.dataset_id, 
        model_key = request.model_key.value,
        task_key = request.task_key.value)
