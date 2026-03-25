import joblib
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from ML.config import (
    ARTIFACTS_DIR,
    DEFAULT_LABEL_COLUMN,
    DEFAULT_MAX_FEATURES,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DEFAULT_TEXT_COLUMN,
    MODEL_PATH,
    RAW_DATA_DIR,
    VECTORIZER_PATH,
)
from ML.models.model_factory import create_text_model

from ML.preprocess import TextPreprocessor


def train_model(
    dataset_path: str,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
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

    print("[2/8] Loading dataset into pandas DataFrame...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    required_columns = {text_column, label_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    print("[3/8] Selecting required columns and dropping empty rows...")
    df = df[[text_column, label_column]].dropna()
    print(f"Rows remaining after cleanup: {len(df)}")

    print("[4/8] Preprocessing text data...")
    preprocessor = TextPreprocessor()
    cleaned_texts = preprocessor.transform(df[text_column])
    print("Text preprocessing completed.")

    print("[5/8] Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts,
        df[label_column],
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=df[label_column],
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
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved vectorizer to: {VECTORIZER_PATH}")
    print(f"Saved model to: {MODEL_PATH}")

    metrics = {
        "dataset_path" : str(data_path),
        "row_count" : len(df),
        "accuracy" : float(accuracy),
        "macro_f1" : float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1" : float(report_dict["weighted avg"]["f1-score"]),
        "classification_report" : report_dict,
        "model_key" : model_key,
        "model_path" : str(MODEL_PATH),
        "vectorizer_path" : str(VECTORIZER_PATH),
    }

    print(f"Training completed using dataset: {data_path.name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(report_text)

    return metrics

if __name__ == "__main__":
    train_model("spam_training_v2.csv", model_key="logistic_regression")

    
