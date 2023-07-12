import random
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame

from arin_privacy_publication.distribution.base_distribution import BaseDistribution


class Csv(BaseDistribution):
    # mean of exponential is 1/ lambda, standard deviation is als 1/ lambda.
    # here we fix the standard deviation to 1/ lambda and shift the distribution to the left to match the desired mean
    def __init__(self, list_mean, list_standard_deviation, path_file_csv, list_series_name: List[str]):
        super().__init__("Csv", list_mean, list_standard_deviation)
        self.path_file_csv = path_file_csv
        self.list_series_name = list_series_name
        dataset_base = pd.read_csv(path_file_csv)
        self.dataset = DataFrame()  # select the series
        for mean, standard_deviation, serie_name_base in zip(list_mean, list_standard_deviation, list_series_name):
            series_name = f"series_{len(self.dataset.columns)}"
            self.dataset[series_name] = dataset_base[serie_name_base]
            # change mean to 0
            self.dataset[series_name] = self.dataset[series_name] - self.dataset[series_name].mean()
            # change std to 0
            self.dataset[series_name] = self.dataset[series_name] / self.dataset[series_name].std()
            # change srd to desired std
            self.dataset[series_name] = self.dataset[series_name] * standard_deviation
            # change mean to desired mean
            self.dataset[series_name] = self.dataset[series_name] + mean

    def _sample(
        self,
        count: int,
        mean: float,
        standard_deviation: float,
    ) -> List[float]:
        raise NotImplementedError()

    def sample(self, count: int) -> DataFrame:
        dataset = self.dataset.sample(count, replace=True).reset_index(drop=True)
        return dataset

    def add(
        self,
        dataset: DataFrame,
    ) -> DataFrame:
        raise NotImplementedError()

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "list_mean": self.list_mean,
            "list_standard_deviation": self.list_standard_deviation,
            "path_file_csv": self.path_file_csv,
            "list_series_name": self.list_series_name,
        }

    @staticmethod
    def from_dict(jsondict: dict) -> "BaseDistribution":
        return Csv(
            jsondict["list_mean"],
            jsondict["list_standard_deviation"],
            jsondict["path_file_csv"],
            jsondict["list_series_name"],
        )
