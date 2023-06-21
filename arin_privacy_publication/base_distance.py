from abc import ABC, abstractmethod
from typing import List


class BaseDistance(ABC):
    """
    Abstract class for divergence measures.
    """

    @abstractmethod
    def __call__(self, a: List[float], b: List[float]) -> float:
        raise NotImplementedError()
