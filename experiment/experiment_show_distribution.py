import random
from typing import List

import matplotlib.pyplot as plt

from arin_privacy_publication.distribution.base_distribution import BaseDistribution
from arin_privacy_publication.distribution.csv import Csv
from arin_privacy_publication.distribution.exponential import Exponential
from arin_privacy_publication.distribution.laplace import Laplace
from arin_privacy_publication.distribution.normal import Normal
from arin_privacy_publication.distribution.uniform import Uniform


# Experiment 1
def experiment_show_distribution(do_run: bool, do_plot: bool, do_show: bool):
    sample_size = 2000000
    list_data_generator: List[BaseDistribution] = [
        Normal([0, 1], [1, 1]),
        Laplace([0, 1], [1, 1]),
        Uniform([0, 1], [1, 1]),
        Exponential([0, 1], [1, 1]),
        Csv([0, 1], [1, 1], "kidney_disease.csv", ["age", "bp"]),
    ]
    if do_run:
        pass
    if do_plot:
        plt.figure(figsize=(8, 7))
        # plt.subplot(1, 5, 1)
        #        plt.ylabel("Relative frequency")
        for i, data_generator in enumerate(list_data_generator):
            plt.subplot(5, 1, i + 1)
            plt.title(data_generator.distribution_name)

            data_frame = data_generator.sample(sample_size)
            plt.hist(
                data_frame[data_frame.columns[0]],
                bins=50,
                label=data_generator.distribution_name,
                density=True,
            )
            #            plt.xlabel("Value")

            plt.xlim(-4, 4)
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.6)
    if do_show:
        plt.show()


if __name__ == "__main__":
    experiment_show_distribution(True, True, True)
