import matplotlib.pyplot as plt

from arin_privacy_publication.distance.min_squared import MinSquared
from arin_privacy_publication.distribution.laplace import Laplace
from arin_privacy_publication.distribution.normal import Normal
from arin_privacy_publication.estimator.max import Max
from arin_privacy_publication.estimator.mean import Mean
from arin_privacy_publication.estimator.multy import Multy
from arin_privacy_publication.estimator.variance import Variance
from arin_privacy_publication.tools_privacy import compute_privacy_dmr


# Experiment 1
def experiment_distribution_invarriance():
    list_r = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    k = 10000  # setting this value to 50000 takes a long time to run (60min+)
    f = MinSquared()
    d = Normal().sample([0], [1], [50])
    list_mean_50 = []
    list_max_50 = []
    list_variance_50 = []
    list_multy_50 = []
    for r in list_r:
        list_mean_50.append(compute_privacy_dmr(k, d, r, f, Mean()))
        list_max_50.append(compute_privacy_dmr(k, d, r, f, Max()))
        list_variance_50.append(compute_privacy_dmr(k, d, r, f, Variance()))
        list_multy_50.append(compute_privacy_dmr(k, d, r, f, Multy([Mean(), Max(), Variance()])))

    plt.figure(figsize=(10, 5))

    plt.plot(list_r, list_mean_50, label="mean_200")
    plt.plot(list_r, list_max_50, label="max_200")
    plt.plot(list_r, list_variance_50, label="variance_200")
    plt.plot(list_r, list_multy_50, label="multy_200")

    plt.xlabel("fraction of D1 in Dr")
    plt.ylabel("fraction of successful attacks")
    plt.title("Privacy DMR")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    experiment_distribution_invarriance()
