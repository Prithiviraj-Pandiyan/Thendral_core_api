from abc import ABC, abstractmethod


class InferenceEngine(ABC):
    """Common contract for all inference services."""

    def __init__(self, task_key: str, model_key: str) -> None:
        self.task_key = task_key
        self.model_key = model_key

    @abstractmethod
    def predict(self, text: str) -> dict:
        """Run inference for a single input record."""
