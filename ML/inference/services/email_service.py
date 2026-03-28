from pathlib import Path

import joblib

from ML.config import ARTIFACTS_DIR, MODEL_PATH, VECTORIZER_PATH
from ML.content_processing import process_content
from ML.core.model_profiles import get_profile
from ML.core.tasks import TaskKey
from ML.inference.engine import InferenceEngine
from ML.preprocessors import get_preprocessor


class EmailInferenceService(InferenceEngine):
    """Inference service for email/text classification tasks."""

    def __init__(self, task_key: str, model_key: str) -> None:
        super().__init__(task_key=task_key, model_key=model_key)
        print(f"[inference] Initializing EmailInferenceService for task={task_key}, model={model_key}")
        self.preprocessor = get_preprocessor(task_key=task_key, input_type="text_email")

        self.spam_profile = get_profile(TaskKey.SPAM_DETECTION.value, model_key)
        self.spam_threshold = self.spam_profile.spam_threshold or 0.5

        self.spam_vectorizer_path, self.spam_model_path = self._resolve_artifacts(
            task_key=TaskKey.SPAM_DETECTION.value,
            model_key=model_key,
        )
        self.ham_vectorizer_path, self.ham_model_path = self._resolve_artifacts(
            task_key=TaskKey.HAM_INTENT.value,
            model_key=model_key,
        )

        print(f"[inference] Loading spam vectorizer: {self.spam_vectorizer_path}")
        self.spam_vectorizer = joblib.load(self.spam_vectorizer_path)
        print(f"[inference] Loading spam model: {self.spam_model_path}")
        self.spam_model = joblib.load(self.spam_model_path)

        self.ham_vectorizer = None
        self.ham_model = None
        if self.ham_vectorizer_path.exists() and self.ham_model_path.exists():
            print(f"[inference] Loading ham-intent vectorizer: {self.ham_vectorizer_path}")
            self.ham_vectorizer = joblib.load(self.ham_vectorizer_path)
            print(f"[inference] Loading ham-intent model: {self.ham_model_path}")
            self.ham_model = joblib.load(self.ham_model_path)

        print("[inference] EmailInferenceService ready.")

    def _resolve_artifacts(self, task_key: str, model_key: str) -> tuple[Path, Path]:
        task_vectorizer = ARTIFACTS_DIR / task_key / model_key / "vectorizer.joblib"
        task_model = ARTIFACTS_DIR / task_key / model_key / "model.joblib"

        # Backward compatibility for old spam artifacts.
        if task_key == TaskKey.SPAM_DETECTION.value and not (task_vectorizer.exists() and task_model.exists()):
            return VECTORIZER_PATH, MODEL_PATH
        return task_vectorizer, task_model

    def _is_spam_label(self, label: object) -> bool:
        normalized = str(label).strip().lower()
        return normalized in {"1", "spam", "true"}

    def _predict_spam_detection(self, text: str) -> dict:
        processed = process_content(text, source_type="email")
        cleaned_text = self.preprocessor.clean_text(processed["normalized_text"])
        spam_vector = self.spam_vectorizer.transform([cleaned_text])
        spam_label = self.spam_model.predict(spam_vector)[0]
        spam_probabilities = self.spam_model.predict_proba(spam_vector)[0]
        spam_confidence = float(max(spam_probabilities))
        is_spam = self._is_spam_label(spam_label) and spam_confidence >= self.spam_threshold

        if is_spam:
            return {
                "input_text": text,
                "detected_html": processed["detected_html"],
                "cleaned_text": cleaned_text,
                "is_spam": True,
                "spam_label": "spam",
                "spam_confidence": spam_confidence,
                "ham_intent": None,
                "ham_intent_confidence": None,
            }

        ham_intent = None
        ham_intent_confidence = None
        if self.ham_vectorizer is not None and self.ham_model is not None:
            ham_vector = self.ham_vectorizer.transform([cleaned_text])
            ham_label = self.ham_model.predict(ham_vector)[0]
            ham_probabilities = self.ham_model.predict_proba(ham_vector)[0]
            ham_intent = str(ham_label)
            ham_intent_confidence = float(max(ham_probabilities))

        return {
            "input_text": text,
            "detected_html": processed["detected_html"],
            "cleaned_text": cleaned_text,
            "is_spam": False,
            "spam_label": "ham",
            "spam_confidence": spam_confidence,
            "ham_intent": ham_intent,
            "ham_intent_confidence": ham_intent_confidence,
        }

    def _predict_ham_intent(self, text: str) -> dict:
        if self.ham_vectorizer is None or self.ham_model is None:
            raise FileNotFoundError(
                f"Ham-intent artifacts not found for model_key={self.model_key}. Train ham_intent first."
            )

        processed = process_content(text, source_type="email")
        cleaned_text = self.preprocessor.clean_text(processed["normalized_text"])
        ham_vector = self.ham_vectorizer.transform([cleaned_text])
        ham_label = self.ham_model.predict(ham_vector)[0]
        ham_probabilities = self.ham_model.predict_proba(ham_vector)[0]
        ham_confidence = float(max(ham_probabilities))
        return {
            "input_text": text,
            "detected_html": processed["detected_html"],
            "cleaned_text": cleaned_text,
            "ham_intent": str(ham_label),
            "ham_intent_confidence": ham_confidence,
        }

    def predict(self, text: str) -> dict:
        if self.task_key == TaskKey.SPAM_DETECTION.value:
            return self._predict_spam_detection(text)
        if self.task_key == TaskKey.HAM_INTENT.value:
            return self._predict_ham_intent(text)
        raise ValueError(f"Unsupported task_key for EmailInferenceService: {self.task_key}")
