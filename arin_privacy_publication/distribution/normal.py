import random
from typing import List

import numpy as np

from arin_privacy_publication.distribution.base_distribution import BaseDistribution


class Normal(BaseDistribution):
    def __init__(self, list_mean: List[float], list_standard_deviation: List[float]):
        super().__init__("Normal", list_mean, list_standard_deviation)

    def _sample(
        self,
        count: int,
        mean: float,
        standard_deviation: float,
    ) -> List[float]:
        return np.random.normal(mean, standard_deviation, count).tolist()

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "list_mean": self.list_mean,
            "list_standard_deviation": self.list_standard_deviation,
        }

    @staticmethod
    def from_dict(jsondict: dict) -> "BaseDistribution":
        return Normal(jsondict["list_mean"], jsondict["list_standard_deviation"])
