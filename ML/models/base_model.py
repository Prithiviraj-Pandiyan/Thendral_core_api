from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTextModel(ABC):
    """Contract for text models used by the training pipeline."""

    @property
    @abstractmethod
    def key(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def build(self) -> Any:
        raise NotImplementedError
