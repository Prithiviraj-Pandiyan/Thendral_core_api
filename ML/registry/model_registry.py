import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ML.config import MODEL_REGISTRY_DIR


logger = logging.getLogger(__name__)


def ensure_model_registry_dir() -> None:
    MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


def register_training_run(dataset_id: str, metrics: dict) -> dict:
    ensure_model_registry_dir()
    run_id = uuid4().hex

    record = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }

    record_path = MODEL_REGISTRY_DIR / f"{run_id}.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("Training run registered: %s (dataset=%s)", run_id, dataset_id)
    return record
