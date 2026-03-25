from sklearn.linear_model import LogisticRegression

from ML.models.base_model import BaseTextModel


class LogisticRegressionTextModel(BaseTextModel):
    def __init__(self, random_state: int, max_iter: int = 1000) -> None:
        self._random_state = random_state
        self._max_iter = max_iter

    @property
    def key(self) -> str:
        return "logistic_regression"

    def build(self) -> LogisticRegression:
        return LogisticRegression(
            max_iter=self._max_iter,
            random_state=self._random_state,
        )
