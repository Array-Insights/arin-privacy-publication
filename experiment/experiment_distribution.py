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
from arin_privacy_publication.estimator.mann_whitney_u_test import MannWhitneyUTest
from arin_privacy_publication.estimator.max import Max
from arin_privacy_publication.estimator.mean import Mean
from arin_privacy_publication.estimator.multy import Multy
from arin_privacy_publication.estimator.variance import Variance
from arin_privacy_publication.estimator.welch_t_test import WelchTTest
from arin_privacy_publication.tools_experiment import create_experiment_dmr, run_experiment
from arin_privacy_publication.tools_privacy import compute_privacy_dmr


# Experiment 1
def experiment_distribution():
    list_reference_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    run_count = 1000  # setting this value to 50000 takes a long time to run (60min+)
    distance = MinSquared()
    sample_size = 200
    ignore_cache = False
    #
    list_data_generator = [
        Normal([0, 1], [5, 5]),
        Laplace([0, 1], [5, 5]),
        Uniform([0, 1], [5, 5]),
        Exponential([0, 1], [5, 5]),
        Csv([0, 1], [5, 5], "kidney_disease.csv", ["age", "bp"]),
    ]
    # list_data_generator = [
    #     Csv([0, 1], [5, 5], "kidney_disease.csv", ["age", "bp"]),
    # ]

    list_estimator = [Mean(), Max(), Variance(), Multy([Mean(), Max(), Variance()]), WelchTTest(), MannWhitneyUTest()]
    # list_estimator = [Max()]

    list_experiment = []
    for estimator in list_estimator:
        for data_generator in list_data_generator:
            experiment = create_experiment_dmr(
                data_generator, sample_size, run_count, distance, estimator, list_reference_rate
            )
            list_experiment.append(experiment)

    random.shuffle(list_experiment)  # shuffle so not all the long experiments are at the end.
    # Improves duration estimation
    for experiment in tqdm(list_experiment):  # TODO make this parallel
        run_experiment(experiment, ignore_cache=ignore_cache)

    # list_estimator = [Mean()]

    # for estimator in list_estimator:
    #     plt.figure(figsize=(10, 5))
    #     for data_generator in list_data_generator:
    #         experiment = create_experiment_dmr(
    #             data_generator, sample_size, run_count, distance, estimator, list_reference_rate
    #         )
    #         result = run_experiment(experiment)
    #         plot_label = estimator.estimator_name + " " + data_generator.distribution_name
    #         plt.plot(list_reference_rate, result["result"]["list_dmr"], label=plot_label)

    #     plt.xlabel("fraction of D1 in Dr")
    #     plt.ylabel("fraction of successful attacks")
    #     plt.title("Effect of distribution on DMR")
    #     plt.xlim(0.0, 1.0)
    #     plt.ylim(0.5, 1.0)
    #     plt.legend()

    # # create plot of bar chart with error bars of dmr_auc
    # plt.figure(figsize=(10, 5))

    list_estimator_name = []
    list_estimator_drm_auc_mean = []
    list_estimator_drm_auc_min = []
    list_estimator_drm_auc_max = []
    for estimator in list_estimator:
        list_dmr_auc = []
        for data_generator in list_data_generator:
            experiment = create_experiment_dmr(
                data_generator, sample_size, run_count, distance, estimator, list_reference_rate
            )
            result = run_experiment(experiment)
            list_dmr_auc.append(result["result"]["dmr_auc"])

        list_estimator_name.append(estimator.estimator_name)
        list_estimator_drm_auc_mean.append(np.mean(list_dmr_auc))
        list_estimator_drm_auc_min.append(np.min(list_dmr_auc))
        list_estimator_drm_auc_max.append(np.max(list_dmr_auc))

    # creating error
    # y_errormin = [0.1, 0.5, 0.9, 0.1, 0.9]
    # y_errormax = [0.2, 0.4, 0.6, 0.4, 0.2]

    y_error = [
        np.array(list_estimator_drm_auc_mean) - np.array(list_estimator_drm_auc_min),
        np.array(list_estimator_drm_auc_max) - np.array(list_estimator_drm_auc_mean),
    ]
    # TODO boxplot?
    # plotting graph
    plt.bar(list_estimator_name, list_estimator_drm_auc_mean)

    plt.errorbar(
        list_estimator_name, list_estimator_drm_auc_mean, yerr=y_error, fmt="o", color="r"
    )  # you can use color ="r" for red or skip to default as blue
    plt.xlabel("Estimator")
    plt.ylabel("DMR AUC")
    plt.ylim(0.5, 1.0)
    plt.show()


if __name__ == "__main__":
    experiment_distribution()
