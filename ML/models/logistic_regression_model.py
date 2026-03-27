from sklearn.linear_model import LogisticRegression

from ML.models.base_model import BaseTextModel


class LogisticRegressionTextModel(BaseTextModel):
    key = "logistic_regression"

    def __init__(
        self,
        random_state: int,
        C: float = 1.0,
        class_weight: str | None = None,
    ) -> None:
        self._random_state = random_state
        self._C = C
        self._class_weight = class_weight

    def build(self) -> LogisticRegression:
        return LogisticRegression(
            max_iter=1000,
            random_state=self._random_state,
            C=self._C,
            class_weight=self._class_weight,
        )
