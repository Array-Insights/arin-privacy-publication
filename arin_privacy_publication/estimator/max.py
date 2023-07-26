from typing import List

import numpy
from pandas import DataFrame

from arin_privacy_publication.estimator.base_estimator import BaseEstimator


class Max(BaseEstimator):
    def __init__(self):
        super().__init__("Max")

    def sensitivity(self, dataset: DataFrame) -> float:
        list_value = sorted(dataset[dataset.columns[0]])
        # differnce between max and second max
        return float(list_value[-1] - list_value[-2])

    def __call__(self, dataset: DataFrame) -> List[float]:
        if 1 < len(dataset):
            return [float(numpy.max(dataset[dataset.columns[0]]))]
        else:
            return [0]

    @staticmethod
    def from_dict(jsondict: dict) -> "BaseEstimator":
        return Max()

    def to_dict(self) -> dict:
        return {"type": self.__class__.__name__}
