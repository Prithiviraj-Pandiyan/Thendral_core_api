import joblib
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from ML.core.tasks import TaskKey

from ML.config import (
    ARTIFACTS_DIR,
    DEFAULT_MAX_FEATURES,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DEFAULT_TEXT_COLUMN,
    RAW_DATA_DIR,
)
from ML.models.model_factory import create_text_model

from ML.preprocess import TextPreprocessor

def _resolve_label_column(task_key:str)->str:
    if task_key ==TaskKey.SPAM_DETECTION.value:
        return "label"
    if task_key == TaskKey.HAM_INTENT.value:
        return "ham_label"
    raise ValueError(f"Unsupported task_key: {task_key}")

def _resolve_artifacts_paths(task_key:str, model_key:str) -> tuple[Path,Path]:
    task_model_dir = ARTIFACTS_DIR/task_key/model_key
    task_model_dir.mkdir(parents=True, exist_ok=True)
    return (task_model_dir/"vectorizer.joblib", task_model_dir/"model.joblib")



def train_model(
    dataset_path: str,
    task_key: str,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str | None = None,
    model_key: Optional[str] = None,
) -> dict:
    if not model_key:
        raise ValueError("model_key is required")
    data_path = Path(dataset_path)

    if not data_path.is_absolute():
        data_path = RAW_DATA_DIR/data_path
    print(f"[1/8] Starting training with dataset: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if not task_key :
        raise ValueError(f"task_key not found")
    if not model_key :
        raise ValueError(f"model_key not found")

    effective_label_column = label_column or _resolve_label_column(task_key)
    vectorizer_path, model_path = _resolve_artifacts_paths(task_key, model_key)

    print("[2/8] Loading dataset into pandas DataFrame...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    required_columns = {text_column, effective_label_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    print("[3/8] Selecting required columns and dropping empty rows...")
    df = df[[text_column, effective_label_column]].dropna()
    print(f"Rows remaining after cleanup: {len(df)}")

    print("[4/8] Preprocessing text data...")
    preprocessor = TextPreprocessor()
    cleaned_texts = preprocessor.transform(df[text_column])
    print("Text preprocessing completed.")

    print("[5/8] Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts,
        df[effective_label_column],
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=df[effective_label_column],
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    print("[6/8] Converting text into TF-IDF vectors...")
    vectorizer = TfidfVectorizer(max_features=DEFAULT_MAX_FEATURES, ngram_range=(1, 2))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    print(f"Vectorization completed with vocabulary size: {len(vectorizer.vocabulary_)}")

    print("[7/8] Building and training model...")
    model_impl = create_text_model(model_key=model_key, random_state=DEFAULT_RANDOM_STATE)
    model = model_impl.build()
    model.fit(X_train_vectorized, y_train)
    print(f"Model training completed using: {model_impl.key}")

    print("[8/8] Evaluating model performance...")
    predictions = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, predictions)
    report_text = classification_report(y_test, predictions)
    report_dict = classification_report(y_test, predictions, output_dict=True)

    print("Saving model artifacts...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)
    print(f"Saved vectorizer to: {vectorizer_path}")
    print(f"Saved model to: {model_path}")

    metrics = {
        "dataset_path" : str(data_path),
        "row_count" : len(df),
        "accuracy" : float(accuracy),
        "macro_f1" : float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1" : float(report_dict["weighted avg"]["f1-score"]),
        "classification_report" : report_dict,
        "model_key" : model_key,
        "task_key" : task_key,
        "model_path" : str(model_path),
        "vectorizer_path" : str(vectorizer_path),
    }

    print(f"Training completed using dataset: {data_path.name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(report_text)

    return metrics

if __name__ == "__main__":
    train_model(
        dataset_path="spam_training_v2.csv", 
        task_key=TaskKey.SPAM_DETECTION.value, 
        model_key="logistic_regression"
        )

    
