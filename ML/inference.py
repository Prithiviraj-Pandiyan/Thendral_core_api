import joblib
from pathlib import Path

from ML.config import ARTIFACTS_DIR, MODEL_PATH, VECTORIZER_PATH
from ML.core.model_profiles import get_profile
from ML.preprocess import TextPreprocessor


class SpamInferenceService:
    def __init__(self) -> None:
        print("[1/5] Initializing inference service...")
        self.preprocessor = TextPreprocessor()
        self.spam_profile = get_profile("spam_detection", "logistic_regression")
        self.spam_threshold = self.spam_profile.spam_threshold or 0.5

        self.spam_vectorizer_path, self.spam_model_path = self._resolve_spam_artifacts()
        self.ham_vectorizer_path = (
            ARTIFACTS_DIR / "ham_intent" / "logistic_regression" / "vectorizer.joblib"
        )
        self.ham_model_path = ARTIFACTS_DIR / "ham_intent" / "logistic_regression" / "model.joblib"

        print(f"[2/5] Loading spam vectorizer: {self.spam_vectorizer_path}")
        self.spam_vectorizer = joblib.load(self.spam_vectorizer_path)
        print(f"[3/5] Loading spam model: {self.spam_model_path}")
        self.spam_model = joblib.load(self.spam_model_path)

        print(f"[4/5] Loading ham-intent vectorizer: {self.ham_vectorizer_path}")
        self.ham_vectorizer = joblib.load(self.ham_vectorizer_path)
        print(f"[5/5] Loading ham-intent model: {self.ham_model_path}")
        self.ham_model = joblib.load(self.ham_model_path)
        print("Inference service is ready.")

    def _resolve_spam_artifacts(self) -> tuple[Path, Path]:
        task_vectorizer = ARTIFACTS_DIR / "spam_detection" / "logistic_regression" / "vectorizer.joblib"
        task_model = ARTIFACTS_DIR / "spam_detection" / "logistic_regression" / "model.joblib"
        if task_vectorizer.exists() and task_model.exists():
            return task_vectorizer, task_model
        return VECTORIZER_PATH, MODEL_PATH

    def _is_spam_label(self, label: object) -> bool:
        normalized = str(label).strip().lower()
        return normalized in {"1", "spam", "true"}

    def predict(self, text: str) -> dict:
        print("Running 2-stage prediction...")
        cleaned_text = self.preprocessor.clean_text(text)

        # Stage 1: spam vs ham
        spam_vector = self.spam_vectorizer.transform([cleaned_text])
        spam_label = self.spam_model.predict(spam_vector)[0]
        spam_probabilities = self.spam_model.predict_proba(spam_vector)[0]
        spam_confidence = float(max(spam_probabilities))
        is_spam = self._is_spam_label(spam_label) and spam_confidence >= self.spam_threshold

        if is_spam:
            return {
                "input_text": text,
                "cleaned_text": cleaned_text,
                "is_spam": True,
                "spam_label": "spam",
                "spam_confidence": spam_confidence,
                "ham_intent": None,
                "ham_intent_confidence": None,
            }

        # Stage 2: ham intent classification
        ham_vector = self.ham_vectorizer.transform([cleaned_text])
        ham_label = self.ham_model.predict(ham_vector)[0]
        ham_probabilities = self.ham_model.predict_proba(ham_vector)[0]
        ham_confidence = float(max(ham_probabilities))

        return {
            "input_text": text,
            "cleaned_text": cleaned_text,
            "is_spam": False,
            "spam_label": "ham",
            "spam_confidence": spam_confidence,
            "ham_intent": str(ham_label),
            "ham_intent_confidence": ham_confidence,
        }

if __name__ == "__main__":
    service = SpamInferenceService()

    sample_text = "Hi Prithvi, we are pleased to offer you the position of Software Engineer."
    result = service.predict(sample_text)

    print("Inference result : ")
    print(result)
