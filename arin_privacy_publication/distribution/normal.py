import random
from typing import List

import numpy as np

from arin_privacy_publication.distribution.base_distribution import BaseDistribution


class Normal(BaseDistribution):
    def __init__(self):
        self.generator_name = "normal distribution"

    def _sample(self, mean: float, standard_deviation: float, count: int) -> List[float]:
        return np.random.normal(mean, standard_deviation, count).tolist()

    def __call__(sample: List[List[float]]):
        return random.normalvariate(0, 1)
