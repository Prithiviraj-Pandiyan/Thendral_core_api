import logging
from pathlib import Path
from uuid import uuid4

from ML.config import DEFAULT_LABEL_COLUMN, DEFAULT_TEXT_COLUMN, PROCESSED_DATA_DIR
from ML.data_sources.huggingface_adapter import load_huggingface_dataframe
from ML.data_sources.upload_adapter import load_dataframe_from_file, save_uploaded_file
from ML.pipelines.validation import normalize_dataframe
from ML.registry.dataset_registry import register_dataset


logger = logging.getLogger(__name__)


def ingest_uploaded_dataset(
    original_file_name: str,
    file_bytes: bytes,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> dict:
    raw_file_path = save_uploaded_file(original_file_name, file_bytes)
    df = load_dataframe_from_file(raw_file_path, text_column=text_column)
    normalized_df = normalize_dataframe(
        df, text_column=text_column, label_column=label_column
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_file_path = PROCESSED_DATA_DIR / f"{uuid4().hex}.csv"
    normalized_df.to_csv(processed_file_path, index=False)
    logger.info("Normalized dataset written: %s", processed_file_path)

    return register_dataset(
        original_file_name=original_file_name,
        raw_file_path=Path(raw_file_path),
        processed_file_path=processed_file_path,
        row_count=len(normalized_df),
    )


def ingest_huggingface_dataset(
    dataset_name: str,
    split: str = "train",
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
    config_name: str | None = None,
    revision: str | None = None,
) -> dict:
    split = split or "train"

    df = load_huggingface_dataframe(
        dataset_name=dataset_name,
        split=split,
        text_column=text_column,
        label_column=label_column,
        config_name=config_name,
        revision=revision,
    )
    normalized_df = normalize_dataframe(
        df,
        text_column=text_column,
        label_column=label_column,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_file_path = PROCESSED_DATA_DIR / f"{uuid4().hex}.csv"
    normalized_df.to_csv(processed_file_path, index=False)
    logger.info("Hugging Face normalized dataset written: %s", processed_file_path)

    return register_dataset(
        original_file_name=dataset_name,
        raw_file_path=f"huggingface://{dataset_name}?split={split}",
        processed_file_path=processed_file_path,
        row_count=len(normalized_df),
    )
