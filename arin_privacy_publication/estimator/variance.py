from typing import List

import numpy
from pandas import DataFrame

from arin_privacy_publication.estimator.base_estimator import BaseEstimator


class Variance(BaseEstimator):
    def __init__(self):
        self.statistic_name = "variance"

    def __call__(self, dataset: DataFrame) -> List[float]:

        if 1 < len(dataset):
            return [float(numpy.var(dataset[dataset.columns[0]]))]
        else:
            return [0]
