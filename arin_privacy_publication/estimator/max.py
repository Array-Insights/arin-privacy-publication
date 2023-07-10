from typing import List

import numpy
from pandas import DataFrame

from arin_privacy_publication.estimator.base_estimator import BaseEstimator


class Max(BaseEstimator):
    def __init__(self):
        self.statistic_name = "max"

    def global_sensitivity(self):
        return 1

    def local_sensitivity(self, dataset: DataFrame):
        return 1

    def __call__(self, dataset: DataFrame) -> List[float]:
        if 1 < len(dataset):
            return [float(numpy.max(dataset[dataset.columns[0]]))]
        else:
            return [0]
