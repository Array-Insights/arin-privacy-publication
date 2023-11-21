import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from arin_privacy_publication.distance.min_squared import MinSquared
from arin_privacy_publication.distribution.normal import Normal
from arin_privacy_publication.estimator.max import Max
from arin_privacy_publication.estimator.mean import Mean
from arin_privacy_publication.estimator.multy import Multy
from arin_privacy_publication.estimator.variance import Variance
from arin_privacy_publication.tools_experiment import (
    create_experiment_dmr,
    run_experiment,
)


# Experiment 1
def experiment_sample_size(
    do_run: bool, do_plot: bool, do_show: bool, do_save: bool
) -> None:
    list_reference_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    run_count = 10000  # setting this value to 50000 takes a long time to run (60min+)
    distance = MinSquared()
    list_sample_size = [
        50,
        54,
        60,
        64,
        70,
        74,
        80,
        84,
        90,
        94,
        100,
        104,
        110,
        120,
        130,
        140,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        800,
        1000,
    ]
    list_sample_size_short = [50, 100, 200, 400]
    list_line_style = ["-", "--", "-.", ":"]
    ignore_cache = False
    #
    data_generator = Normal([0, 1], [1, 1])
    list_estimator = [Mean(), Max(), Variance(), Multy([Mean(), Max(), Variance()])]
    # list_estimator = [Mean()]

    if do_run:
        list_experiment = []
        for estimator in list_estimator:
            for sample_size in list_sample_size:
                experiment = create_experiment_dmr(
                    data_generator,
                    sample_size,
                    run_count,
                    distance,
                    estimator,
                    list_reference_rate,
                )
                list_experiment.append(experiment)

        random.shuffle(
            list_experiment
        )  # shuffle so not all the long experiments are at the end.
        # Improves duration estimation
        for experiment in tqdm(list_experiment):  # TODO make this parallel
            run_experiment(experiment, ignore_cache)

    # list_estimator = [Mean()]
    if do_plot:
        plt.figure(figsize=(5, 4))
        plt.title("Effect of sample size on DMR")

        for i, estimator in enumerate(list_estimator):
            plt.subplot(2, 2, i + 1)
            plt.title(estimator.estimator_name)
            for sample_size, line_style in zip(list_sample_size_short, list_line_style):
                experiment = create_experiment_dmr(
                    data_generator,
                    sample_size,
                    run_count,
                    distance,
                    estimator,
                    list_reference_rate,
                )
                result = run_experiment(experiment)
                plot_label = " N=" + str(sample_size)
                plt.plot(
                    list_reference_rate,
                    result["result"]["list_dmr"],
                    label=plot_label,
                    linestyle=line_style,
                )
                plt.subplots_adjust(
                    left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.5
                )

                plt.legend()
                plt.xlabel("fraction of X1 in Xr")
                plt.ylabel("fraction of successful attacks")

                plt.xlim(0.0, 1.0)
                plt.ylim(0.5, 1.0)

    if do_save:
        plt.savefig("figure/sample_size_various.png", dpi=300, bbox_inches="tight")

    if do_plot:
        # plot the results
        plt.figure(figsize=(5, 4))
        for estimator in list_estimator:
            list_auc_dmr = []
            for sample_size in list_sample_size:
                experiment = create_experiment_dmr(
                    data_generator,
                    sample_size,
                    run_count,
                    distance,
                    estimator,
                    list_reference_rate,
                )
                result = run_experiment(experiment)
                list_auc_dmr.append(result["result"]["dmr_auc"])
            plot_label = estimator.estimator_name

            # Sample size (number of trials)
            n_trials = 100

            # Compute the 95% confidence interval analytically
            # using the Wilson score interval method.
            z = 1.96  # Critical value for a 95% confidence interval
            array_auc_dmr = np.array(list_auc_dmr)

            lower_bound = list_auc_dmr - z * np.sqrt(
                (array_auc_dmr * (1 - array_auc_dmr) + z**2 / (4 * run_count))
                / run_count
            )
            upper_bound = list_auc_dmr + z * np.sqrt(
                (array_auc_dmr * (1 - array_auc_dmr) + z**2 / (4 * run_count))
                / run_count
            )
            array_auc_dmr_err = np.array(
                [[array_auc_dmr - lower_bound], [upper_bound - array_auc_dmr]]
            )
            array_auc_dmr_err = np.squeeze(array_auc_dmr_err)
            # plt.errorbar(list_sample_size, list_auc_dmr, yerr=array_auc_dmr_err, fmt="o", capsize=5)

            plt.errorbar(list_sample_size, list_auc_dmr, label=plot_label)

        plt.xlabel("Sample size")
        plt.ylabel("DMR AUC")
        plt.title("Effect of sample size on DMR")
        plt.legend(loc="upper right")

    if do_save:
        plt.savefig("figure/sample_size.png", dpi=300, bbox_inches="tight")

    if do_show:
        plt.show()


if __name__ == "__main__":
    experiment_sample_size(True, True, True, True)
