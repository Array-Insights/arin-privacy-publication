import random
from typing import List

import numpy as np


class Normal:
    def __init__(self):
        self.generator_name = "normal distribution"

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

    def __call__(sample: List[List[float]]):
        return random.normalvariate(0, 1)
