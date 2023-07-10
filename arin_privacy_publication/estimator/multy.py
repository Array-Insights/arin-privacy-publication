from typing import List

import numpy
from pandas import DataFrame

from arin_privacy_publication.estimator.base_estimator import BaseEstimator


class Multy(BaseEstimator):
    def __init__(self, list_statistic: List[BaseEstimator]):
        self.statistic_name = "multy"
        self.list_statistic = list_statistic

    def __call__(self, dataset: DataFrame) -> List[float]:
        list_result = []
        for statistic in self.list_statistic:
            list_result.extend(statistic(dataset))
        return list_result
