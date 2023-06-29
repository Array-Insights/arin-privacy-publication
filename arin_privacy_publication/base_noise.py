from abc import ABC, abstractmethod
from typing import List


class BaseNoise(ABC):

    # Abstract class for noise generators.

    # std_multiplier is the standard deviation multiplier
    def __init__(self, std_multiplier: float):
        self.std_multiplier = std_multiplier

    @abstractmethod
    def __call__(self, sample: List[List[float]]) -> List[List[float]]:
        raise NotImplementedError()
