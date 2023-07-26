from typing import List

import scipy
from pandas import DataFrame

from arin_privacy_publication.distribution.laplace import Laplace
from arin_privacy_publication.estimator.base_estimator import BaseEstimator


class StudentTTest(BaseEstimator):
    def __init__(self, alternative: str = "less", epsilon: float = 0, return_p: bool = True):
        super().__init__("Student T-test")
        # TODO epsilon 0 is infitite noise epsilon infinite is zero noise
        self.alternative = alternative
        self.epsilon = epsilon
        self.return_p = return_p
        if epsilon < 0:
            raise ValueError("epsilon must be equal to or greater than 0")

    def sensitivity(self, dataset: DataFrame) -> float:
        raise NotImplementedError()

    def __call__(self, dataset: DataFrame) -> List[float]:
        if 0 < self.epsilon:
            sdev_0 = dataset[dataset.columns[0]].std()
            sdev_1 = dataset[dataset.columns[1]].std()
            distribution = Laplace([0, 0], [sdev_0 * self.epsilon, sdev_1 * self.epsilon])
            dataset = distribution.add(dataset)

        result = scipy.stats.ttest_ind(
            dataset[dataset.columns[0]],
            dataset[dataset.columns[1]],
            equal_var=True,
            alternative=self.alternative,
        )

        if self.return_p:
            return result
        else:
            return result[0]

    @staticmethod
    def from_dict(jsondict: dict) -> "BaseEstimator":
        return StudentTTest(
            alternative=jsondict["alternative"], epsilon=jsondict["epsilon"], return_p=jsondict["return_p"]
        )

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "alternative": self.alternative,
            "epsilon": self.epsilon,
            "return_p": self.return_p,
        }
