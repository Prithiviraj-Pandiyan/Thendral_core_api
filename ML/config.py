from pathlib import Path

# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "ML" / "artifacts"

# Artifact file paths
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer.joblib"
MODEL_PATH = ARTIFACTS_DIR / "logistic_regression_model.joblib"
LABELS_PATH = ARTIFACTS_DIR / "labels.joblib"

# Training config
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_LABEL_COLUMN = "label"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_FEATURES = 5000
