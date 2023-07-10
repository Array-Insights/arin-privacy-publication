from abc import ABC, abstractmethod
from typing import List

from pandas import DataFrame


class BaseDistribution(ABC):

    # Abstract class for noise generators.

    # std_multiplier is the standard deviation multiplier
    def __init__(self, distribution_name: str, list_mean: List[float], list_standard_deviation: List[float]):
        self.distribution_name = distribution_name
        self.list_mean = list_mean
        self.list_standard_deviation = list_standard_deviation

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def _sample(
        self,
        count: int,
        mean: float,
        standard_deviation: float,
    ) -> List[float]:
        raise NotImplementedError()

    def sample(self, count: int) -> DataFrame:
        dataset = DataFrame()
        for mean, standard_deviation in zip(self.list_mean, self.list_standard_deviation):
            sample = self._sample(count, mean, standard_deviation)
            dataset[f"series_{len(dataset)}"] = sample
        return dataset

    def add(
        self,
        dataset: DataFrame,
    ) -> DataFrame:
        dataset = dataset.copy()
        for i, column in enumerate(dataset.columns):
            dataset[column] = dataset[column] + self._sample(
                len(dataset[column]), self.list_mean[i], self.list_standard_deviation[i]
            )
        return dataset

    @staticmethod
    def from_dict(jsondict: dict) -> "BaseDistribution":

        # from arin_privacy_publication.distribution.uniform import Uniform
        from arin_privacy_publication.distribution.laplace import Laplace
        from arin_privacy_publication.distribution.no_noise import NoNoise
        from arin_privacy_publication.distribution.normal import Normal

        if jsondict["type"] == Normal.__name__:
            return Normal.from_dict(jsondict)
        elif jsondict["type"] == NoNoise.__name__:
            return NoNoise.from_dict(jsondict)
        elif jsondict["type"] == Laplace.__name__:
            return Laplace.from_dict(jsondict)
        else:
            raise ValueError(f"Unknown distribution name: {jsondict['type']}")
