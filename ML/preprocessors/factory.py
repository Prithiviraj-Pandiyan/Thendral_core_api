from ML.preprocessors.base import BasePreprocessor
from ML.preprocessors.text_email_preprocessor import TextEmailPreprocessor


def get_preprocessor(
    task_key: str | None = None,
    input_type: str = "text_email",
    preprocessor_key: str | None = None,
) -> BasePreprocessor:
    """
    Return a preprocessor instance.

    Current behavior:
    - text/email tasks use TextEmailPreprocessor
    - `task_key` is accepted now so routing can expand cleanly later
    """
    selected_key = preprocessor_key or input_type

    if selected_key == "text_email":
        return TextEmailPreprocessor()

    raise ValueError(
        f"Unsupported preprocessor selection: task_key={task_key}, input_type={input_type}, preprocessor_key={preprocessor_key}"
    )
