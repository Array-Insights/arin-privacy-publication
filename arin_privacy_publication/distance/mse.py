from typing import List

import numpy

from arin_privacy_publication.distance.base_distance import BaseDistance


class Mse(BaseDistance):
    def __init__(self):
        pass

    def __call__(self, a: List[float], b: List[float]) -> float:
        return float(numpy.mean((numpy.array(a) - numpy.array(b)) ** 2))
