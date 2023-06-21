from typing import List

import numpy

from arin_privacy_publication.base_statistic import BaseStatistic


class VarianceStatistic(BaseStatistic):
    def __init__(self):
        self.statistic_name = "variance"

    def __call__(self, sample: List[float]) -> List[float]:
        return [float(numpy.var(sample))]
