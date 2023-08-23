import matplotlib
import matplotlib.pyplot as plt
import scienceplots
from experiment_distribution import experiment_distribution
from experiment_power import experiment_power
from experiment_sample_size import experiment_sample_size
from experiment_sensitivity import experiment_sensitivity

plt.style.use(["science", "ieee"])
matplotlib.rcParams["text.usetex"] = False
# matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# experiment_sensitivity(True, True, False, True)
# experiment_sample_size(True, True, False, True)
experiment_power(True, True, False, True)
experiment_distribution(True, True, False, True)

# plt.show()
