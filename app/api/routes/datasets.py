from fastapi import APIRouter, Depends, File, UploadFile
from app.services.auth import require_admin_key

from app.schemas.dataset import DatasetTrainingRequest, HuggingFaceIngestRequest
from app.services.dataset_service import (
    ingest_huggingface_dataset_service,
    list_datasets_service,
    train_dataset_service,
    upload_dataset_service,
)

router = APIRouter()


@router.post("/upload", dependencies=[Depends(require_admin_key)])
async def upload_file(file: UploadFile = File(...)):
    return await upload_dataset_service(file)


@router.post("/huggingface", dependencies=[Depends(require_admin_key)])
def upload_huggingface_dataset(request: HuggingFaceIngestRequest):
    return ingest_huggingface_dataset_service(
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
    return train_dataset_service(request.dataset_id, request.model_key)
