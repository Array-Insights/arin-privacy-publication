from typing import List

import numpy as np

from arin_privacy_publication.distribution.base_distribution import BaseDistribution


class Laplace(BaseDistribution):
    def __init__(self):
        super().__init__("laplace")

    def _sample(
        self,
        mean: List[float],
        standard_deviation: List[float],
        count: List[int],
    ) -> List[float]:

        return np.random.laplace(mean, standard_deviation / np.sqrt(2), count)

    def __call__(self, sample: List[List[float]]) -> List[List[float]]:
        for i in range(len(sample)):
            noise = np.random.laplace(0.0, self.std_multiplier / np.sqrt(2), len(sample[i]))
            sample[i] = (np.array(sample[i]) + noise).tolist()

        return sample
