from enum import Enum

from pydantic import BaseModel


class ModelKey(str, Enum):
    logistic_regression = "logistic_regression"


class DatasetTrainingRequest(BaseModel):
    dataset_id: str
    model_key: ModelKey


class HuggingFaceIngestRequest(BaseModel):
    dataset_name: str
    split: str = "train"
    text_column: str = "text"
    label_column: str = "label"
    config_name: str | None = None
    revision: str | None = None
