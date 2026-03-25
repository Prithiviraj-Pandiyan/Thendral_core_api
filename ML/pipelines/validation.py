import pandas as pd

from ML.config import DEFAULT_LABEL_COLUMN, DEFAULT_TEXT_COLUMN


def normalize_dataframe(
    df: pd.DataFrame,
    text_column: str = DEFAULT_TEXT_COLUMN,
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> pd.DataFrame:
    normalized_df = df.copy()

    if text_column not in normalized_df.columns:
        raise ValueError(f"Missing required text column: {text_column}")

    if label_column not in normalized_df.columns:
        normalized_df[label_column] = 0

    normalized_df = normalized_df[[text_column, label_column]].dropna()
    normalized_df[text_column] = normalized_df[text_column].astype(str)
    normalized_df[label_column] = normalized_df[label_column].astype(int)
    return normalized_df
