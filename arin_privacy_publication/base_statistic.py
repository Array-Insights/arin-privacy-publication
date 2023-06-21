from abc import ABC, abstractmethod
from typing import List


class BaseStatistic(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, sampel: List[float]) -> List[float]:
        raise NotImplementedError()
