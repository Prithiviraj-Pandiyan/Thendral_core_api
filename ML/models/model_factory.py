from ML.models.logistic_regression_model import LogisticRegressionTextModel

MODEL_BUILDERS = {
    "logistic_regression": LogisticRegressionTextModel,
}


def create_text_model(model_key: str, random_state: int, **kwargs):
    model_cls = MODEL_BUILDERS.get(model_key)
    if not model_cls:
        raise ValueError(
            f"Unsupported model_key='{model_key}'. "
            f"Supported: {sorted(MODEL_BUILDERS.keys())}"
        )
    return model_cls(random_state=random_state, **kwargs)


def get_supported_model_keys() -> list[str]:
    return sorted(MODEL_BUILDERS.keys())
