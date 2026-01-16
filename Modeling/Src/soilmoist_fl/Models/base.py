# Jakob Balkovec
# Model Base

from Modeling.Utils.logging import get_logger


class BaseModel:
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.log = get_logger("models")

    def name(self):
        return self.__class__.__name__

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
