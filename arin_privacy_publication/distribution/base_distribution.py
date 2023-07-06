from abc import ABC, abstractmethod
from typing import List


class BaseDistribution(ABC):

    # Abstract class for noise generators.

    # std_multiplier is the standard deviation multiplier
    def __init__(self, distribution_name: str):
        self.distribution_name = distribution_name

    @abstractmethod
    def _sample(self, mean: float, standard_deviation: float, count: int) -> List[float]:
        raise NotImplementedError()

    def sample(
        self, list_mean: List[float], list_standard_deviation: List[float], list_count: List[int]
    ) -> List[List[float]]:
        sample = []
        for mean, standard_deviation, count in zip(list_mean, list_standard_deviation, list_count):
            sample.append(self._sample(mean, standard_deviation, count))
        return sample

    @abstractmethod
    def __call__(
        self,
        sample: List[List[float]],
    ) -> List[List[float]]:
        raise NotImplementedError()
