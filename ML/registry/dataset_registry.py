import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ML.config import DATASET_REGISTRY_DIR


logger = logging.getLogger(__name__)


def ensure_dataset_registry_dir() -> None:
    DATASET_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


def register_dataset(
    original_file_name: str,
    raw_file_path: str | Path,
    processed_file_path: Path,
    row_count: int,
) -> dict:
    ensure_dataset_registry_dir()
    dataset_id = processed_file_path.stem

    record = {
        "dataset_id": dataset_id,
        "original_file_name": original_file_name,
        "raw_file_path": str(raw_file_path),
        "processed_file_path": str(processed_file_path),
        "row_count": row_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_training",
    }

    registry_path = DATASET_REGISTRY_DIR / f"{dataset_id}.json"
    registry_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("Dataset registered: %s (%s rows)", dataset_id, row_count)
    return record


def list_registered_datasets() -> list[dict]:
    ensure_dataset_registry_dir()
    records: list[dict] = []

    for file_path in DATASET_REGISTRY_DIR.glob("*.json"):
        records.append(json.loads(file_path.read_text(encoding="utf-8")))

    records.sort(key=lambda item: item["created_at"], reverse=True)
    return records


def get_dataset_record(dataset_id: str) -> dict:
    registry_path = DATASET_REGISTRY_DIR / f"{dataset_id}.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_id}")
    return json.loads(registry_path.read_text(encoding="utf-8"))
