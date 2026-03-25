import logging

from fastapi import HTTPException, UploadFile

from ML.models.model_factory import get_supported_model_keys
from ML.pipelines.ingestion import ingest_huggingface_dataset, ingest_uploaded_dataset
from ML.pipelines.training import train_with_dataset_record
from ML.registry.dataset_registry import get_dataset_record, list_registered_datasets


logger = logging.getLogger(__name__)


def _build_ingest_response(record: dict, task_key: str) -> dict:
    return {
        **record,
        "task_key": task_key,
        "recommended_train_payload": {
            "dataset_id": record["dataset_id"],
            "task_key": task_key,
            "model_key": "logistic_regression",
        },
    }


def _resolve_label_column(task_key: str, label_column: str | None) -> str:
    cleaned = (label_column or "").strip()
    if cleaned:
        return cleaned
    normalized_task_key = str(task_key).strip().lower()
    if normalized_task_key.endswith("ham_intent"):
        return "ham_label"
    return "label"


async def upload_dataset_service(
    file: UploadFile,
    task_key: str,
    text_column: str,
    label_column: str,
) -> dict:
    resolved_label_column = _resolve_label_column(task_key, label_column)
    logger.info(
        "Dataset upload requested: file=%s task_key=%s text_column=%s label_column=%s",
        file.filename,
        task_key,
        text_column,
        resolved_label_column,
    )
    file_bytes = await file.read()

    try:
        record = ingest_uploaded_dataset(
            original_file_name=file.filename or "uploaded_file",
            file_bytes=file_bytes,
            task_key=task_key,
            text_column=text_column,
            label_column=resolved_label_column,
        )
    except ValueError as exc:
        logger.warning("Dataset upload failed validation: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("Dataset upload completed: %s", record["dataset_id"])
    return _build_ingest_response(record, task_key=task_key)


def ingest_huggingface_dataset_service(
    dataset_name: str,
    task_key: str,
    split: str,
    text_column: str,
    label_column: str,
    config_name: str | None = None,
    revision: str | None = None,
) -> dict:
    resolved_label_column = _resolve_label_column(task_key, label_column)
    logger.info(
        "Hugging Face ingestion requested: dataset=%s task_key=%s config=%s split=%s label_column=%s",
        dataset_name,
        task_key,
        config_name,
        split,
        resolved_label_column,
    )
    try:
        record = ingest_huggingface_dataset(
            task_key=task_key,
            dataset_name=dataset_name,
            split=split,
            text_column=text_column,
            label_column=resolved_label_column,
            config_name=config_name,
            revision=revision,
        )
    except (ValueError, RuntimeError) as exc:
        message = str(exc)

        # Make network/proxy/firewall failures human-readable
        if (
            "WinError 10013" in message
            or "Cannot send a request, as the client has been closed" in message
            or "Failed to establish a new connection" in message
            or "ConnectionError" in message
            or "timeout" in message.lower()
        ):
            detail = {
                "message": "Hugging Face access failed due to network/firewall/proxy restrictions.",
                "how_to_fix": [
                    "Allow outbound HTTPS for python to huggingface.co and cdn-lfs.huggingface.co",
                    "If you are behind a proxy, set HTTP_PROXY and HTTPS_PROXY",
                    "Retry the request after network policy update",
                ],
                "raw_error": message,
            }
            logger.warning("Hugging Face ingestion blocked by network policy: %s", message)
            raise HTTPException(status_code=503, detail=detail) from exc

        logger.warning("Hugging Face ingestion failed: %s", message)
        raise HTTPException(status_code=400, detail=message) from exc


    logger.info("Hugging Face ingestion completed: %s", record["dataset_id"])
    return _build_ingest_response(record, task_key=task_key)


def list_datasets_service() -> list[dict]:
    records = list_registered_datasets()
    logger.info("Listed %s registered datasets", len(records))
    return records


def train_dataset_service(dataset_id: str, task_key: str, model_key: str) -> dict:
    supported_model_keys = get_supported_model_keys()
    if model_key not in supported_model_keys:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Invalid model_key '{model_key}'",
                "supported_models": supported_model_keys,
            },
        )

    logger.info(
        "Training requested for dataset_id=%s using model_key=%s and task_key=%s",
        dataset_id,
        model_key,
        task_key,
    )
    try:
        record = get_dataset_record(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        result = train_with_dataset_record(record, task_key=task_key, model_key=model_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "Training completed for dataset_id=%s using model_key=%s and task_key=%s with accuracy=%.4f",
        dataset_id,
        model_key,
        task_key,
        result["training_metrics"]["accuracy"],
    )
    return {
        "status": "trained",
        "task_key": task_key,
        "dataset": {
            "dataset_id": result["dataset"]["dataset_id"],
            "original_file_name": result["dataset"]["original_file_name"],
            "row_count": result["dataset"]["row_count"],
        },
        "model": {
            "model_key": result["training_metrics"]["model_key"],
            "run_id": result["model_run"]["run_id"],
            "created_at": result["model_run"]["created_at"],
        },
        "metrics": {
            "accuracy": result["training_metrics"]["accuracy"],
            "macro_f1": result["training_metrics"]["macro_f1"],
            "weighted_f1": result["training_metrics"]["weighted_f1"],
        },
        "artifacts": {
            "model_path": result["training_metrics"]["model_path"],
            "vectorizer_path": result["training_metrics"]["vectorizer_path"],
        },
    }
