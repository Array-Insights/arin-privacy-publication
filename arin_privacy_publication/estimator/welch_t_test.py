from typing import List

import scipy
from pandas import DataFrame

from arin_privacy_publication.distribution.laplace import Laplace
from arin_privacy_publication.estimator.base_estimator import BaseEstimator


class WelchTTest(BaseEstimator):
    def __init__(self, alternative: str = "less", epsilon: float = 0):
        super().__init__("Welch T-test")
        self.alternative = alternative
        self.epsilon = epsilon
        if epsilon < 0:
            raise ValueError("epsilon must be equal to or greater than 0")

    def __call__(self, dataset: DataFrame) -> List[float]:
        if 0 < self.epsilon:
            sdev_0 = dataset[dataset.columns[0]].std()
            sdev_1 = dataset[dataset.columns[1]].std()
            distribution = Laplace([0, 0], [sdev_0 * self.epsilon, sdev_1 * self.epsilon])
            dataset = distribution.add(dataset)

        return scipy.stats.ttest_ind(
            dataset[dataset.columns[0]],
            dataset[dataset.columns[1]],
            equal_var=False,
            alternative=self.alternative,
        )

    @staticmethod
    def from_dict(jsondict: dict) -> "BaseEstimator":
        return WelchTTest(alternative=jsondict["alternative"], epsilon=jsondict["epsilon"])

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "alternative": self.alternative,
            "epsilon": self.epsilon,
        }
