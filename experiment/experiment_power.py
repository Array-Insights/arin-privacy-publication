import matplotlib.pyplot as plt
import numpy as np
import scipy
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
from arin_privacy_publication.tools_privacy import (
    compute_power,
    compute_privacy_dmr,
    create_experiment_power,
    run_experiment,
)


# Experiment 1
def experiment_power():

    print(scipy.stats.ttest_ind([0, 1, 2], [2, 3, 4], equal_var=True))
    sample_size = 200

    list_effect_size = np.arange(-0.2, 0.5, 0.02)
    list_epsilon = [0.0, 0.5]
    run_count = 5000
    plt.figure()
    list_legend = []
    for epsilon in list_epsilon:
        list_test = [StudentTTest(epsilon=epsilon), WelchTTest(epsilon=epsilon), MannWhitneyUTest(epsilon=epsilon)]
        list_list_power = []
        for test in list_test:
            list_power = []
            for effect_size in tqdm(list_effect_size):
                data_generator = Normal([0, effect_size], [1, 1])
                experiment = create_experiment_power(test, data_generator, sample_size, 0.05, run_count)
                result = run_experiment(experiment)
                power = result["power_mean"]
                list_power.append(power)
            list_list_power.append(list_power)

        # plt.style.use(["science", "ieee"])

        for list_power in list_list_power:
            plt.plot(list_effect_size, list_power)
        plt.xlabel("Effect size (in sdev)")
        plt.ylabel("Power")
        list_legend.append(f"Student's t-test, epsilon={epsilon}")
        list_legend.append(f"Welch's t-test epsilon={epsilon}")
        list_legend.append(f"Mann-Whitney U-test epsilon={epsilon}")
    plt.legend(list_legend)
    plt.show()


if __name__ == "__main__":
    experiment_power()
