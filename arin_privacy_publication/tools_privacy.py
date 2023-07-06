import random
from typing import List, Optional

import numpy as np

from arin_privacy_publication.distance.base_distance import BaseDistance
from arin_privacy_publication.distribution.base_distribution import BaseDistribution
from arin_privacy_publication.distribution.no_noise import NoNoise
from arin_privacy_publication.distribution.normal import Normal
from arin_privacy_publication.estimator.base_estimator import BaseEstimator
from arin_privacy_publication.estimator.student_t_test import StudentTTestStatistic


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


def compute_power(
    test: StudentTTestStatistic,
    distribution: BaseDistribution,
    effect_size: float,
    sample_size: int,
    signigicance: float = 0.05,
    run_count: int = 1000,
) -> float:

    success_count = 0
    # For each row in our column
    for i in range(run_count):
        # scale noise to the resolution size for this column
        test_result = test(distribution.sample([0, effect_size], [1, 1], [sample_size, sample_size]))
        if test_result[1] < signigicance:
            success_count += 1
    return success_count / run_count
