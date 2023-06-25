from typing import List

import numpy as np

from arin_privacy_publication.base_statistic import BaseStatistic


class VarianceStatistic(BaseStatistic):
    def __init__(self):
        self.statistic_name = "variance"

    def __call__(self, sample: List[float]) -> List[float]:
        return [float(np.var(sample))]

class SafeVarianceStatistic(BaseStatistic):
    def __init__(self, epsilon=0.5):
        self.statistic_name = "variance"
        self.epsilon = epsilon

    def __call__(self, sample: List[float]) -> List[float]:
        # Sensitivity of the variance function:
        mean = np.mean(sample)
        max_diff = np.max(sample) - np.min(sample)
        sensitivity = max_diff**2 / len(sample)


        beta = sensitivity / self.epsilon
        noisy_sumsq = np.sum(np.square(sample)) + np.random.laplace(0, beta)
        noisy_sum = np.sum(sample) + np.random.laplace(0, beta)
        noisy_count = len(sample) + np.random.laplace(0, beta)

        noisy_variance = (noisy_sumsq / noisy_count) - np.square(noisy_sum / noisy_count)

        return noisy_variance
