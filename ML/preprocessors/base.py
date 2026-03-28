from abc import ABC, abstractmethod
from typing import Iterable, List


class BasePreprocessor(ABC):
    """Base contract for all preprocessors."""

    @abstractmethod
    def clean_text(self, text: str) -> str:
        """Clean a single input record."""

    def transform(self, texts: Iterable[str]) -> List[str]:
        """Clean multiple input records."""
        return [self.clean_text(text) for text in texts]
