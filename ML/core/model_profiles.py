from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingProfile:
    max_features: int
    ngram_min: int
    ngram_max: int
    class_weight: str | None
    C: float
    spam_threshold: float | None  # only used for spam_detection inference


DEFAULT_PROFILE = TrainingProfile(
    max_features=5000,
    ngram_min=1,
    ngram_max=2,
    class_weight=None,
    C=1.0,
    spam_threshold=None,
)

PROFILES: dict[tuple[str, str], TrainingProfile] = {
    ("spam_detection", "logistic_regression"): TrainingProfile(
        max_features=8000,
        ngram_min=1,
        ngram_max=2,
        class_weight="balanced",
        C=1.0,
        spam_threshold=0.80,
    ),
    ("ham_intent", "logistic_regression"): TrainingProfile(
        max_features=8000,
        ngram_min=1,
        ngram_max=2,
        class_weight="balanced",
        C=1.0,
        spam_threshold=None,
    ),
}


def get_profile(task_key: str, model_key: str) -> TrainingProfile:
    return PROFILES.get((task_key, model_key), DEFAULT_PROFILE)
