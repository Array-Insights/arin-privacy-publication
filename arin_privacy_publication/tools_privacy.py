import json
import os
import random
from hashlib import sha256
from typing import List, Optional

import numpy as np

from arin_privacy_publication.distance.base_distance import BaseDistance
from arin_privacy_publication.distribution.base_distribution import BaseDistribution
from arin_privacy_publication.distribution.no_noise import NoNoise
from arin_privacy_publication.distribution.normal import Normal
from arin_privacy_publication.estimator.base_estimator import BaseEstimator
from arin_privacy_publication.estimator.student_t_test import StudentTTest


def compute_privacy_dmr(
    k: int,
    d: List[List[float]],
    r: float,
    f: BaseDistance,
    s: BaseEstimator,
    noise: Optional[BaseDistribution] = None,
) -> float:
    """
    Compute the privacy DMR between two distributions.
    """
    if len(d) == 0:
        raise ValueError("d must not be empty")
    if len(d[0]) % 2 != 0:
        raise ValueError("d must have an even number of elements")
    if r < 0 or r > 1:
        raise ValueError("r must be in [0, 1]")
    if k <= 0:
        raise ValueError("k must be positive")

    if noise is None:
        noise = NoNoise()

    n = len(d[0]) // 2
    dr_size = int(r * n)
    if dr_size == 0:
        return 0.5
    succes_count = 0
    for _ in range(k):
        print("here")
        random.shuffle(d)
        d1 = d[:n]
        d2 = d[n:]
        dr = d1[:dr_size]  # because we shuffle d, we can just take the first dr_size elements
        if f(s(noise(d1)), s(dr)) < f(s(noise(d2)), s(dr)):
            succes_count += 1
            print("here")
    return succes_count / k


def add_laplace_noise(sample: List[float], scale: float = 1.0, resolution: Optional[float] = None) -> List[float]:

    sample = sample.copy()
    # find std_dev to help us scale our noise addition
    std_dev = np.std(sample)
    # Generate our distribution
    noise = np.random.laplace(0.0, 0.1 * std_dev * scale, 10000)
    # For each row in our column
    for i in range(len(sample)):
        # scale noise to the resolution size for this column
        if resolution is not None:
            noise[i] = noise[i] - (noise[i] % resolution)
        # Add the noise to the column
        sample[i] += noise[i]
    return sample


def sha256_dict(dict: dict) -> str:
    return sha256(json.dumps(dict).encode()).hexdigest()


def run_experiment(dict_experiment: dict) -> dict:
    if not os.path.isdir("experiment_cache"):
        os.mkdir("experiment_cache")
    experiment_hash = sha256_dict(dict_experiment)
    path_file_experiment = os.path.join("experiment_cache", experiment_hash + ".json")
    if os.path.exists(path_file_experiment):
        with open(path_file_experiment, "r") as file:
            return json.load(file)
    else:
        experiment_type = dict_experiment["experiment_type"]
        if experiment_type == "power":
            result = run_experiment_power(dict_experiment)
        else:
            raise ValueError("Unknown experiment type {experiment_type}")
        with open(path_file_experiment, "w") as file:
            json.dump(result, file)
        return result


def create_experiment_power(
    test: BaseEstimator,
    data_generator: BaseDistribution,
    sample_size: int,
    signigicance: float,
    run_count: int,
) -> dict:
    experiment = {}
    experiment["experiment_type"] = "power"
    experiment["test"] = test.to_dict()
    experiment["data_generator"] = data_generator.to_dict()
    experiment["sample_size"] = sample_size
    experiment["signigicance"] = signigicance
    experiment["run_count"] = run_count

    return experiment


def run_experiment_power(experiment: dict) -> dict:
    test: BaseEstimator = BaseEstimator.from_dict(experiment["test"])
    data_generator: BaseDistribution = BaseDistribution.from_dict(experiment["data_generator"])
    sample_size: int = experiment["sample_size"]
    signigicance: float = experiment["signigicance"]
    run_count: int = experiment["run_count"]

    power_mean = compute_power(test, data_generator, sample_size, signigicance, run_count)
    result = {}
    result["power_mean"] = power_mean
    return result


def compute_power(
    test: BaseEstimator,
    data_generator: BaseDistribution,
    sample_size: int,
    signigicance: float = 0.05,
    run_count: int = 1000,
) -> float:

    success_count = 0
    # For each row in our column
    for i in range(run_count):
        # scale noise to the resolution size for this column
        test_result = test(data_generator.sample(sample_size))
        if test_result[1] < signigicance:
            success_count += 1
    return success_count / run_count
