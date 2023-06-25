from typing import List

import numpy

from arin_privacy_publication.base_statistic import BaseStatistic


class MaxStatistic(BaseStatistic):
    def __init__(self):
        self.statistic_name = "max"

    def __call__(self, sample: List[float]) -> List[float]:
        return [float(numpy.max(sample))]


class SafeMaxStatistic(BaseStatistic):
    def __init__(self, epsilon=0.5):
        self.statistic_name = "dp_max"
        self.epsilon = epsilon

    def __call__(self, sample: List[float]) -> List[float]:
        # Sensitivity of the max function:
        # the largest value in the population minus
        # the size of the population
        min_value = min(sample)
        max_value = max(sample)
        sensitivity = max_value - min_value

        beta = sensitivity / self.epsilon
        noisy_max = numpy.max(sample) + numpy.random.laplace(0, beta)

        return noisy_max
