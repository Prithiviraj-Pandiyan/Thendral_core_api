import logging

import pandas as pd
from settings import settings


logger = logging.getLogger(__name__)


def load_huggingface_dataframe(
    dataset_name: str,
    split: str = "train",
    text_column: str = "text",
    label_column: str = "label",
    config_name: str | None = None,
    revision: str | None = None,
) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face support requires `datasets`. Install with: pip install datasets"
        ) from exc

    logger.info(
        "Loading Hugging Face dataset: %s config=%s split=%s revision=%s",
        dataset_name,
        config_name,
        split,
        revision,
    )
    dataset = load_dataset(
        path=dataset_name,
        name=config_name,
        split=split,
        revision=revision,
        token=settings.hf_token,
    )
    df = pd.DataFrame(dataset)

    if text_column not in df.columns:
        raise ValueError(f"Missing text column in Hugging Face data: {text_column}")
    if label_column not in df.columns:
        raise ValueError(f"Missing label column in Hugging Face data: {label_column}")

    return df[[text_column, label_column]].copy()
