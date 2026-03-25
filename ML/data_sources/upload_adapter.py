import logging
from pathlib import Path
from uuid import uuid4

import pandas as pd

from ML.config import RAW_UPLOADS_DIR


logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".txt"}


def ensure_upload_dirs() -> None:
    RAW_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(file_name: str, file_bytes: bytes) -> Path:
    ensure_upload_dirs()
    stored_name = f"{uuid4().hex}_{Path(file_name).name}"
    raw_file_path = RAW_UPLOADS_DIR / stored_name
    raw_file_path.write_bytes(file_bytes)
    logger.info("Uploaded file saved: %s", raw_file_path)
    return raw_file_path


def load_dataframe_from_file(file_path: Path, text_column: str) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".xlsx":
        return pd.read_excel(file_path)

    lines = file_path.read_text(encoding="utf-8").splitlines()
    rows = [line.strip() for line in lines if line.strip()]
    return pd.DataFrame({text_column: rows})
