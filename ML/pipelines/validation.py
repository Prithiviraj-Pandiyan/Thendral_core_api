import pandas as pd
from ML.core.email_labels import HamIntentLabel
from ML.core.tasks import TaskKey
from ML.config import DEFAULT_LABEL_COLUMN, DEFAULT_TEXT_COLUMN


def _validate_spam_labels(series: pd.Series) -> None:
    unique_values = set(series.dropna().tolist())
    allowed = {0, 1, 2, "0", "1", "2", "ham", "spam", "phish"}
    if not unique_values.issubset(allowed):
        raise ValueError(
            f"Invalid spam_detection labels found: {sorted(unique_values)}. "
            "Allowed: 0/1/2 or ham/spam/phish"
        )


def _validate_ham_intent_labels(series: pd.Series) -> None:
    unique_values = set(series.dropna().astype(str).tolist())
    allowed = {label.value for label in HamIntentLabel}
    if not unique_values.issubset(allowed):
        invalid = sorted(unique_values - allowed)
        raise ValueError(
            f"Invalid ham_intent labels found: {invalid}. Allowed: {sorted(allowed)}"
        )

def validate_labels_for_task(df: pd.DataFrame, task_key: str, label_column: str) -> None:
    if label_column not in df.columns:
        raise ValueError(f"Missing required label column: {label_column}")

    if task_key == TaskKey.SPAM_DETECTION.value:
        _validate_spam_labels(df[label_column])
        return

    if task_key == TaskKey.HAM_INTENT.value:
        _validate_ham_intent_labels(df[label_column])
        return

    raise ValueError(f"Unsupported task_key: {task_key}")


def normalize_dataframe(
    df: pd.DataFrame,
    task_key: str,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame:
    normalized_df = df.copy()

    if text_column not in normalized_df.columns:
        raise ValueError(f"Missing required text column: {text_column}")

    if label_column not in normalized_df.columns:
        raise ValueError(f"Missing required label column: {label_column}")

    normalized_df = normalized_df[[text_column, label_column]].dropna()
    normalized_df[text_column] = normalized_df[text_column].astype(str)
    validate_labels_for_task(
        normalized_df,
        task_key=task_key,
        label_column=label_column,
    )

    if task_key == TaskKey.SPAM_DETECTION.value:
        # Accept numeric/text labels and collapse phishing into spam for binary training.
        normalized_df[label_column] = (
            normalized_df[label_column]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(
                {
                    "ham": 0,
                    "spam": 1,
                    "phish": 1,
                    "0": 0,
                    "1": 1,
                    "2": 1,
                }
            )
            .astype(int)
        )
    else:
        normalized_df[label_column] = normalized_df[label_column].astype(str)

    return normalized_df
