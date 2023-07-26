from typing import List

import numpy
from pandas import DataFrame

from arin_privacy_publication.estimator.base_estimator import BaseEstimator


class Variance(BaseEstimator):
    def __init__(self):
        super().__init__("Variance")

    def sensitivity(self, dataset: DataFrame) -> float:
        list_value = sorted(dataset[dataset.columns[0]])
        # the max increas in varriance is by droppign the min or the max (whichever is more distant from the mean)
        # this will shift the mean up but also remove our most extreme error
        # it is easier to try this out than to derive the equation (it would not look very nice anyway)
        var_current = numpy.var(list_value)
        var_change_min = float(abs(var_current - numpy.var(list_value[1:])))
        var_change_max = float(abs(var_current - numpy.var(list_value[:-1])))
        return max([var_change_min, var_change_max])

    def __call__(self, dataset: DataFrame) -> List[float]:

        if 1 < len(dataset):
            return [float(numpy.var(dataset[dataset.columns[0]]))]
        else:
            return [0]

    @staticmethod
    def from_dict(jsondict: dict) -> "BaseEstimator":
        return Variance()

    def to_dict(self) -> dict:
        return {"type": self.__class__.__name__}
