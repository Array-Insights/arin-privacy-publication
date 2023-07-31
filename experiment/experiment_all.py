import matplotlib.pyplot as plt
from experiment_distribution import experiment_distribution
from experiment_power import experiment_power
from experiment_sample_size import experiment_sample_size
from experiment_sensitivity import experiment_sensitivity

experiment_sensitivity(True, True, False)
experiment_power(True, True, False)
experiment_distribution(True, True, False)
experiment_sample_size(True, True, False)
plt.show()
