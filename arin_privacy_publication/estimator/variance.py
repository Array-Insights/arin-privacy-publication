from typing import List

import numpy

from arin_privacy_publication.estimator.base_estimator import BaseEstimator


class Variance(BaseEstimator):
    def __init__(self):
        self.statistic_name = "variance"

    def __call__(self, sample: List[List[float]]) -> List[float]:
        return [float(numpy.var(sample[0]))]
