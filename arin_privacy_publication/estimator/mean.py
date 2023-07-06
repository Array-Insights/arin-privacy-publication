from typing import List

import numpy

from arin_privacy_publication.estimator.base_estimator import BaseEstimator


class Mean(BaseEstimator):
    def __init__(self):
        super().__init__("mean")

    def __call__(self, sample: List[List[float]]) -> List[float]:
        return [float(numpy.mean(sample[0]))]
