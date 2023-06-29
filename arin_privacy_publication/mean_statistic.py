from typing import List

import numpy

from arin_privacy_publication.base_statistic import BaseStatistic


class MeanStatistic(BaseStatistic):
    def __init__(self):
        self.statistic_name = "mean"

    def __call__(self, sample: List[List[float]]) -> List[float]:
        return [float(numpy.mean(sample[0]))]
