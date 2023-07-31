import random

import matplotlib.pyplot as plt
from tqdm import tqdm

from arin_privacy_publication.distance.min_squared import MinSquared
from arin_privacy_publication.distribution.laplace import Laplace
from arin_privacy_publication.distribution.normal import Normal
from arin_privacy_publication.estimator.mann_whitney_u_test import MannWhitneyUTest
from arin_privacy_publication.estimator.max import Max
from arin_privacy_publication.estimator.mean import Mean
from arin_privacy_publication.estimator.multy import Multy
from arin_privacy_publication.estimator.student_t_test import StudentTTest
from arin_privacy_publication.estimator.variance import Variance
from arin_privacy_publication.estimator.welch_t_test import WelchTTest
from arin_privacy_publication.tools_experiment import create_experiment_dmr, create_experiment_power, run_experiment


# Experiment 1
def experiment_power(do_run: bool, do_plot: bool, do_show: bool):

    sample_size = 200
    effect_size = 0.3
    list_epsilon = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3, 3.5, 4.0]
    list_reference_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    run_count = 1000
    data_generator = Normal([0, effect_size], [1, 1])
    distance = MinSquared()
    if do_run:
        list_experiment = []
        for epsilon in list_epsilon:
            list_test = [StudentTTest(epsilon=epsilon), MannWhitneyUTest(epsilon=epsilon)]
            for test in list_test:
                experiment = create_experiment_power(test, data_generator, sample_size, 0.05, run_count)
                list_experiment.append(experiment)

                experiment = create_experiment_dmr(
                    data_generator, sample_size, run_count, distance, test, list_reference_rate
                )
                list_experiment.append(experiment)
        random.shuffle(list_experiment)  # shuffle so not all the long experiments are at the end.
        # Improves duration estimation
        for experiment in tqdm(list_experiment):  # TODO make this parallel
            run_experiment(experiment)
    if do_plot:
        plt.figure()
        list_test = [StudentTTest(epsilon=0), MannWhitneyUTest(epsilon=0)]
        for test in list_test:
            list_power = []
            list_dmr_auc = []
            for epsilon in list_epsilon:
                test.epsilon = epsilon
                experiment = create_experiment_power(test, data_generator, sample_size, 0.05, run_count)
                result = run_experiment(experiment)
                power_mean = result["result"]["power_mean"]
                experiment = create_experiment_dmr(
                    data_generator, sample_size, run_count, distance, test, list_reference_rate
                )
                result = run_experiment(experiment)
                dmr_auc = result["result"]["dmr_auc"]

                list_power.append(power_mean)
                list_dmr_auc.append(dmr_auc)

            plt.plot(list_epsilon, list_power, label=(test.estimator_name + " power"))
            plt.plot(list_epsilon, list_dmr_auc, label=(test.estimator_name + " DMR auc"))
        plt.xlabel("epsilon")
        plt.ylabel("Power")
        plt.legend()
    if do_show:
        plt.show()


if __name__ == "__main__":
    experiment_power(True, True, True)
