from abc import ABC, abstractmethod
from typing import List


class BaseDistance(ABC):
    """
    Abstract class for distance measures.
    """

    @abstractmethod
    def __call__(self, a: List[float], b: List[float]) -> float:
        raise NotImplementedError()

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError()

    @staticmethod
    def from_dict(jsondict: dict) -> "BaseDistance":

        from arin_privacy_publication.distance.mean_squared import MeanSquared
        from arin_privacy_publication.distance.min_squared import MinSquared

        distance_type = jsondict["type"]
        if distance_type == MinSquared.__name__:
            return MinSquared.from_dict(jsondict)
        elif distance_type == MeanSquared.__name__:
            return MeanSquared.from_dict(jsondict)
        else:
            raise ValueError(f"Unknown disteance type: {distance_type}")
