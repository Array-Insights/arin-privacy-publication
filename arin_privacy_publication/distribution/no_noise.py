from typing import List

import numpy as np

from arin_privacy_publication.distribution.base_distribution import BaseDistribution


class NoNoise(BaseDistribution):
    def __init__(self):
        super().__init__("no noise")

    def _sample(self, mean: float, standard_deviation: float, count: int) -> List[float]:
        return np.zeros(count).tolist()

    def __call__(self, sample: List[List[float]]) -> List[List[float]]:
        return sample
