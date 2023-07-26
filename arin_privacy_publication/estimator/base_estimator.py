from abc import ABC, abstractmethod
from typing import List

from pandas import DataFrame


class BaseEstimator(ABC):

    # Abstract class for statistics.

    def __init__(self, estimator_name: str):
        self.estimator_name = estimator_name

    @abstractmethod
    def sensitivity(self, dataset: DataFrame) -> float:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, dataset: DataFrame) -> List[float]:
        raise NotImplementedError()

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError()

    @staticmethod
    def from_dict(jsondict: dict) -> "BaseEstimator":

        from arin_privacy_publication.estimator.mann_whitney_u_test import MannWhitneyUTest
        from arin_privacy_publication.estimator.max import Max
        from arin_privacy_publication.estimator.mean import Mean
        from arin_privacy_publication.estimator.multy import Multy
        from arin_privacy_publication.estimator.student_t_test import StudentTTest
        from arin_privacy_publication.estimator.variance import Variance
        from arin_privacy_publication.estimator.welch_t_test import WelchTTest

        if jsondict["type"] == Mean.__name__:
            return Mean.from_dict(jsondict)
        elif jsondict["type"] == Variance.__name__:
            return Variance.from_dict(jsondict)
        elif jsondict["type"] == Multy.__name__:
            return Multy.from_dict(jsondict)
        elif jsondict["type"] == Max.__name__:
            return Max.from_dict(jsondict)
        elif jsondict["type"] == StudentTTest.__name__:
            return StudentTTest.from_dict(jsondict)
        elif jsondict["type"] == WelchTTest.__name__:
            return WelchTTest.from_dict(jsondict)
        elif jsondict["type"] == MannWhitneyUTest.__name__:
            return MannWhitneyUTest.from_dict(jsondict)
        else:
            raise ValueError(f"Unknown estimator name: {jsondict['estimator_name']}")
