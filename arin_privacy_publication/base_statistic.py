from abc import ABC, abstractmethod
from typing import List


class BaseStatistic(ABC):

    # Abstract class for statistics.

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, sample: List[List[float]]) -> List[float]:
        raise NotImplementedError()
