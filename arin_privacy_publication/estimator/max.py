from typing import List

import numpy

from arin_privacy_publication.estimator.base_estimator import BaseEstimator


class Max(BaseEstimator):
    def __init__(self):
        self.statistic_name = "max"

    def globel_sensitivity(self):
        return 1

    def localglobel_sensitivity(self, sample: List[List[float]]):
        return 1

    def __call__(self, sample: List[List[float]]) -> List[float]:
        if 1 < len(sample):
            return [float(numpy.max(sample[0]))]
        else:
            return [0]
