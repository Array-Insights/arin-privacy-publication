from typing import List

import numpy

from arin_privacy_publication.estimator.base_estimator import BaseEstimator


class Max(BaseEstimator):
    def __init__(self):
        self.statistic_name = "max"

    def sensitivity(self):
        #return local dp sentivity

    def __call__(self, sample: List[List[float]]) -> List[float]:
        return [float(numpy.max(sample[0]))]
