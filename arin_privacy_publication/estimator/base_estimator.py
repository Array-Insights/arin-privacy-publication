from abc import ABC, abstractmethod
from typing import List


class BaseEstimator(ABC):

    # Abstract class for statistics.

    def __init__(self, estimator_name: str):
        self.estimator_name = estimator_name

    @abstractmethod
    def __call__(self, sample: List[List[float]]) -> List[float]:
        raise NotImplementedError()
