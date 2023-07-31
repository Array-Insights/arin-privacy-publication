import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from arin_privacy_publication.distance.min_squared import MinSquared
from arin_privacy_publication.distribution.csv import Csv
from arin_privacy_publication.distribution.exponential import Exponential
from arin_privacy_publication.distribution.laplace import Laplace
from arin_privacy_publication.distribution.normal import Normal
from arin_privacy_publication.distribution.uniform import Uniform
from arin_privacy_publication.estimator.base_estimator import BaseEstimator
from arin_privacy_publication.estimator.max import Max
from arin_privacy_publication.estimator.mean import Mean
from arin_privacy_publication.estimator.multy import Multy
from arin_privacy_publication.estimator.variance import Variance
from arin_privacy_publication.tools_experiment import (
    create_experiment_dmr,
    create_experiment_sensitivity,
    run_experiment,
)


def experiment_sensitivity(do_run: bool, do_plot: bool, do_show: bool) -> None:
    list_reference_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    run_count = 1000  # setting this value to 50000 takes a long time to run (60min+)
    distance = MinSquared()
    sample_size = 200
    ignore_cache = False
    list_data_generator = [
        Normal([0, 1], [5, 5]),
        Laplace([0, 1], [5, 5]),
        Uniform([0, 1], [5, 5]),
        Exponential([0, 1], [5, 5]),
        Csv([0, 1], [5, 5], "kidney_disease.csv", ["age", "bp"]),
    ]
    list_estimator = [Mean(), Max(), Variance()]
    if do_run:
        list_experiment = []
        for data_generator in list_data_generator:
            for estimator in list_estimator:
                experiment = create_experiment_dmr(
                    data_generator, sample_size, run_count, distance, estimator, list_reference_rate
                )
                list_experiment.append(experiment)
                experiment = create_experiment_sensitivity(data_generator, sample_size, run_count, estimator)
                list_experiment.append(experiment)

        random.shuffle(list_experiment)  # shuffle so not all the long experiments are at the end.
        # Improves duration estimation
        for experiment in tqdm(list_experiment):  # TODO make this parallel
            run_experiment(experiment, ignore_cache=ignore_cache)

    if do_plot:
        plt.figure()
        plt.title("sensitivity")
        for data_generator in list_data_generator:
            list_dmr_auc_mean = []
            list_dmr_auc_sdev = []
            list_sensitivty_mean = []
            list_sensitivty_sdev = []

            for estimator in list_estimator:

                experiment = create_experiment_dmr(
                    data_generator, sample_size, run_count, distance, estimator, list_reference_rate
                )
                result = run_experiment(experiment)
                list_dmr_auc_mean.append(result["result"]["dmr_auc"])
                # list_dmr_auc_sdev.append(result["result"]["dmr_auc_sdev"])
                list_dmr_auc_sdev.append(0)
                x = result["result"]["dmr_auc"]

                experiment = create_experiment_sensitivity(data_generator, sample_size, run_count, estimator)
                result = run_experiment(experiment)
                list_sensitivty_mean.append(result["result"]["sensitivity_mean"])
                list_sensitivty_sdev.append(result["result"]["sensitivity_sdev"])

                y = result["result"]["sensitivity_mean"]
                txt = f"{estimator.estimator_name}"
                plt.annotate(txt, (x, y))

            plt.scatter(list_dmr_auc_mean, list_sensitivty_mean, label=data_generator.distribution_name)

        # creating erro
        # y_errormin = [0.1, 0.5, 0.9, 0.1, 0.9]
        # y_errormax = [0.2, 0.4, 0.6, 0.4, 0.2]
        plt.xlabel("DMR AUC")
        plt.ylabel("Sensitivity")
        plt.legend()

    if do_show:
        plt.show()


if __name__ == "__main__":
    experiment_sensitivity(True, True, True)
