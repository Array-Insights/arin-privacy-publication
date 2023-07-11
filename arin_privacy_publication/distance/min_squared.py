from typing import List

import numpy

from arin_privacy_publication.distance.base_distance import BaseDistance


class MinSquared(BaseDistance):
    def __init__(self):
        pass

    def __call__(self, a: List[float], b: List[float]) -> float:
        return float(numpy.min((numpy.array(a) - numpy.array(b)) ** 2))

    @staticmethod
    def from_dict(jsondict: dict) -> "BaseDistance":
        return MinSquared()

    def to_dict(self) -> dict:
        return {"type": self.__class__.__name__}
