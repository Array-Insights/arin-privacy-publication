from typing import List

import numpy
from pandas import DataFrame

from arin_privacy_publication.estimator.base_estimator import BaseEstimator


class Multy(BaseEstimator):
    def __init__(self, list_estimator: List[BaseEstimator]):
        super().__init__("Multy")
        self.list_estimator = list_estimator

    def sensitivity(self, dataset: DataFrame) -> float:
        # sum of all estimator sensitivities
        sum_sensitivity = 0
        for estimator in self.list_estimator:
            sum_sensitivity += estimator.sensitivity(dataset)
        return sum_sensitivity

    def __call__(self, dataset: DataFrame) -> List[float]:
        list_result = []
        for estimator in self.list_estimator:
            list_result.extend(estimator(dataset))
        return list_result

    @staticmethod
    def from_dict(jsondict: dict) -> "BaseEstimator":
        list_estimator = []
        for estimator in jsondict["list_estimator"]:
            list_estimator.append(BaseEstimator.from_dict(estimator))
        return Multy(list_estimator)

    def to_dict(self) -> dict:
        list_estimator = []
        for estimator in self.list_estimator:
            list_estimator.append(estimator.to_dict())
        return {"type": self.__class__.__name__, "list_estimator": list_estimator}
