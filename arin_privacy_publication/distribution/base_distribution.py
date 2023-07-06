from abc import ABC, abstractmethod
from typing import List


class BaseDistribution(ABC):

    # Abstract class for noise generators.

    # std_multiplier is the standard deviation multiplier
    def __init__(self, std_multiplier: float):
        self.std_multiplier = std_multiplier

    @abstractmethod
    def sample(
        self,
        list_mean: List[float],
        list_standard_deviation: List[float],
        list_count: List[int],
    ) -> List[List[float]]:
        raise NotImplementedError()

    @abstractmethod
    def __call__(
        self,
        sample: List[List[float]],
    ) -> List[List[float]]:
        raise NotImplementedError()
