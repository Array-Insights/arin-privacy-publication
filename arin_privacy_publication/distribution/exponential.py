import random
from typing import List

import numpy as np

from arin_privacy_publication.distribution.base_distribution import BaseDistribution


class Exponential(BaseDistribution):
    # mean of exponential is 1/ lambda, standard deviation is als 1/ lambda.
    # here we fix the standard deviation to 1/ lambda and shift the distribution to the left to match the desired mean
    def __init__(self, list_mean: List[float], list_standard_deviation: List[float]):
        super().__init__("Exponential", list_mean, list_standard_deviation)

    def _sample(
        self,
        count: int,
        mean: float,
        standard_deviation: float,
    ) -> List[float]:
        shift = mean - standard_deviation
        return (np.random.exponential(standard_deviation, count) + shift).tolist()

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "list_mean": self.list_mean,
            "list_standard_deviation": self.list_standard_deviation,
        }

    @staticmethod
    def from_dict(jsondict: dict) -> "BaseDistribution":
        return Exponential(jsondict["list_mean"], jsondict["list_standard_deviation"])
