import random

import matplotlib.pyplot as plt
from tqdm import tqdm

from arin_privacy_publication.distance.min_squared import MinSquared
from arin_privacy_publication.distribution.laplace import Laplace
from arin_privacy_publication.distribution.normal import Normal
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
def experiment_sample_size():
    list_reference_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    run_count = 1000  # setting this value to 50000 takes a long time to run (60min+)
    distance = MinSquared()
    list_sample_size = [50, 100, 150, 200, 250, 300, 400, 500, 600]
    #
    data_generator = Normal([0, 1], [1, 1])
    # list_estimator = [Mean(), Max(), Variance(), Multy([Mean(), Max(), Variance()])]
    list_estimator = [Mean()]

    list_experiment = []
    for estimator in list_estimator:
        for sample_size in list_sample_size:
            experiment = create_experiment_dmr(
                data_generator, sample_size, run_count, distance, estimator, list_reference_rate
            )
            list_experiment.append(experiment)

    random.shuffle(list_experiment)  # shuffle so not all the long experiments are at the end.
    # Improves duration estimation
    for experiment in tqdm(list_experiment):  # TODO make this parallel
        run_experiment(experiment)

    # list_estimator = [Mean()]
    plt.figure(figsize=(10, 5))
    for estimator in list_estimator:
        for sample_size in list_sample_size:
            experiment = create_experiment_dmr(
                data_generator, sample_size, run_count, distance, estimator, list_reference_rate
            )
            result = run_experiment(experiment)
            plot_label = estimator.estimator_name + " N=" + str(sample_size)
            plt.plot(list_reference_rate, result["result"]["list_dmr"], label=plot_label)

    plt.xlabel("fraction of D1 in Dr")
    plt.ylabel("fraction of successful attacks")
    plt.title("Effect of sample size on DMR")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.5, 1.0)
    plt.legend()
    # plt.show()

    # prepare the experiments
    list_experiment = []
    for estimator in list_estimator:
        list_auc_dmr = []
        for sample_size in tqdm(list_sample_size):
            experiment = create_experiment_dmr(
                data_generator, sample_size, run_count, distance, estimator, list_reference_rate
            )
            list_experiment.append(experiment)

    random.shuffle(list_experiment)  # shuffle so not all the long experiments are at the end.
    # Improves duration estimation
    for experiment in tqdm(list_experiment):  # TODO make this parallel
        run_experiment(experiment)

    # plot the results
    plt.figure(figsize=(10, 5))
    for estimator in list_estimator:
        list_auc_dmr = []
        for sample_size in list_sample_size:
            experiment = create_experiment_dmr(
                data_generator, sample_size, run_count, distance, estimator, list_reference_rate
            )
            result = run_experiment(experiment)
            list_auc_dmr.append(result["result"]["dmr_auc"])
        plot_label = estimator.estimator_name
        plt.plot(list_sample_size, list_auc_dmr, label=plot_label)

    plt.xlabel("Sample size")
    plt.ylabel("DMR AUC")
    plt.title("Effect of sample size on DMR")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    experiment_sample_size()
