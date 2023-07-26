from typing import List

import numpy
from pandas import DataFrame

from arin_privacy_publication.estimator.base_estimator import BaseEstimator


class Mean(BaseEstimator):
    def __init__(self):
        super().__init__("Mean")

    def sensitivity(self, dataset: DataFrame) -> float:
        list_value = sorted(dataset[dataset.columns[0]])
        # max value  divided by the number of elements
        return list_value[-1] / len(list_value)

    def __call__(self, dataset: DataFrame) -> List[float]:
        if 1 < len(dataset):
            return [float(numpy.mean(dataset[dataset.columns[0]]))]
        else:
            return [0]

    @staticmethod
    def from_dict(jsondict: dict) -> "BaseEstimator":
        return Mean()

    def to_dict(self) -> dict:
        return {"type": self.__class__.__name__}
