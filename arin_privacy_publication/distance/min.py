from typing import List

import numpy

from arin_privacy_publication.base_distance import BaseDistance


class Min(BaseDistance):
    def __init__(self):
        pass

    def __call__(self, a: List[float], b: List[float]) -> float:
        return float(numpy.min((numpy.array(a) - numpy.array(b)) ** 2))
