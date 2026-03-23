import re
from typing import Iterable, List

class TextPreprocessor:
    """
    Common reusable text preprocessing class for Thendral.
    This file should remain model-agnostic so it can be reused for:
    - Logistic Regression
    - Linear Regression (if text-derived numerical features are used)
    - Neural Networks
    - Future transformer pipelines
    """

    def __init__(self, lowercase:bool =True):
        self.lowercase = lowercase

    def clean_text (self, text:str)->str:
        """
        Clean one text string.
        """
        if text is None:
            return ""

        text = str(text)

        if self.lowercase:
            text = text.lower()

        # Remove urls
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Remove email addresses
        text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)

        # Remove non-alphanumeric characters except spaces
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

        # Collapse extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def transform(self, texts: Iterable[str]) -> List[str]:
        """
        Clean multiple text records.
        """
        return [self.clean_text(text) for text in texts]

