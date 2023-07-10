from typing import List

import numpy as np

from arin_privacy_publication.distribution.base_distribution import BaseDistribution


class NoNoise(BaseDistribution):
    def __init__(self):
        super().__init__("no noise", [], [])

    def _sample(
        self,
        count: int,
        mean: float,
        standard_deviation: float,
    ) -> List[float]:
        return np.zeros(count).tolist()

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
        }

    @staticmethod
    def from_dict(jsondict: dict) -> "BaseDistribution":
        return NoNoise()
