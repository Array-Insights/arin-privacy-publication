from typing import List

import numpy
import scipy

from arin_privacy_publication.estimator.base_estimator import BaseEstimator


class StudentTTestStatistic(BaseEstimator):
    def __init__(self, alternative: str = "less"):
        self.statistic_name = "t-test"
        self.alternative = alternative

    def __call__(self, sample: List[List[float]]) -> List[float]:
        return scipy.stats.ttest_ind(sample[0], sample[1], equal_var=True, alternative=self.alternative)
