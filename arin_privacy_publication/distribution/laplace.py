from typing import List

import numpy as np

from arin_privacy_publication.base_distribution import BaseDistribution


class Laplace(BaseDistribution):
    def __init__(self, std_multiplier: float):
        super().__init__(std_multiplier)

    def sample(
        self,
        list_mean: List[float],
        list_standard_deviation: List[float],
        list_count: List[int],
    ) -> List[List[float]]:
        list_sample = []
        for mean, standard_deviation, count in zip(list_mean, list_standard_deviation, list_count):
            list_sample.append(np.random.normal(mean, standard_deviation, count).tolist())
        return list_sample

    def __call__(self, sample: List[List[float]]) -> List[List[float]]:
        for i in range(len(sample)):
            noise = np.random.laplace(0.0, self.std_multiplier / np.sqrt(2), len(sample[i]))
            sample[i] = (np.array(sample[i]) + noise).tolist()

        return sample
