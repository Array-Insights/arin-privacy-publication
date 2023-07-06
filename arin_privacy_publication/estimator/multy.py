from typing import List

import numpy

from arin_privacy_publication.estimator.base_estimator import BaseEstimator


class Multy(BaseEstimator):
    def __init__(self, list_statistic: List[BaseEstimator]):
        self.statistic_name = "multy"
        self.list_statistic = list_statistic

    def __call__(self, sample: List[List[float]]) -> List[float]:
        list_result = []
        for statistic in self.list_statistic:
            list_result.extend(statistic(sample))
        return list_result
