from typing import List

import numpy as np

from arin_privacy_publication.distribution.base_distribution import BaseDistribution


class Laplace(BaseDistribution):
    def __init__(self, list_mean: List[float], list_standard_deviation: List[float]):
        super().__init__("laplace", list_mean, list_standard_deviation)

    def _sample(
        self,
        count: int,
        mean: float,
        standard_deviation: float,
    ) -> List[float]:
        return np.random.laplace(mean, standard_deviation / np.sqrt(2), count).tolist()

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "list_mean": self.list_mean,
            "list_standard_deviation": self.list_standard_deviation,
        }

    @staticmethod
    def from_dict(jsondict: dict) -> "BaseDistribution":
        return Laplace(jsondict["list_mean"], jsondict["list_standard_deviation"])
