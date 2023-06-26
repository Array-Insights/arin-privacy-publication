from typing import List

import numpy as np

from arin_privacy_publication.base_statistic import BaseStatistic


class MeanStatistic(BaseStatistic):
    def __init__(self):
        self.statistic_name = "mean"

    def __call__(self, sample: List[float]) -> List[float]:
        return [float(np.mean(sample))]


class SafeMeanStatistic(BaseStatistic):
    def __init__(self, epsilon=0.5):
        self.statistic_name = "dp_mean"
        self.epsilon = epsilon

    def __call__(self, sample: List[float]) -> List[float]:
        # Sensitivity of the mean function:
        # the largest value in the population divided by
        # the size of the population
        max_abs_value = np.max(sample)
        min_abs_value = np.min(sample)

        sensitivity = (max_abs_value - min_abs_value) / len(sample)

        beta = sensitivity / self.epsilon
        noisy_sum = np.sum(sample) + np.random.laplace(0, beta)
        noisy_count = len(sample) + np.random.laplace(0, beta)

        noisy_mean = noisy_sum / noisy_count

        return noisy_mean
