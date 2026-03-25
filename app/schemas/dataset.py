from enum import Enum
from ML.core.tasks import TaskKey
from pydantic import BaseModel


class ModelKey(str, Enum):
    logistic_regression = "logistic_regression"


class LabelColumnKey(str, Enum):
    label = "label"
    ham_label = "ham_label"


class DatasetTrainingRequest(BaseModel):
    dataset_id: str
    task_key : TaskKey
    model_key: ModelKey


class HuggingFaceIngestRequest(BaseModel):
    dataset_name: str
    task_key: str
    split: str = "train"
    text_column: str = "text"
    label_column: str = "label"
    config_name: str | None = None
    revision: str | None = None
