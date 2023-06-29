from typing import List

import numpy

from arin_privacy_publication.base_statistic import BaseStatistic


class MaxStatistic(BaseStatistic):
    def __init__(self):
        self.statistic_name = "max"

    def __call__(self, sample: List[List[float]]) -> List[float]:
        return [float(numpy.max(sample[0]))]
