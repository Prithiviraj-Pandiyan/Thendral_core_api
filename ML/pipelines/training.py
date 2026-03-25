import logging

from ML.registry.model_registry import register_training_run
from ML.train import train_model


logger = logging.getLogger(__name__)


def train_with_dataset_record(
    record: dict,
    model_key: str,
) -> dict:
    dataset_id = record["dataset_id"]
    metrics = train_model(record["processed_file_path"], model_key=model_key)
    run_record = register_training_run(dataset_id=dataset_id, metrics=metrics)
    logger.info(
        "Training pipeline completed for dataset_id=%s using model_key=%s",
        dataset_id,
        model_key,
    )

    return {
        "dataset": record,
        "training_metrics": metrics,
        "model_run": run_record,
    }
