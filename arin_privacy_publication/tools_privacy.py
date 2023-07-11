import json
import os
import random
from hashlib import sha256
from typing import List, Optional

import numpy as np
from pandas import DataFrame

from arin_privacy_publication.distance.base_distance import BaseDistance
from arin_privacy_publication.distribution.base_distribution import BaseDistribution
from arin_privacy_publication.estimator.base_estimator import BaseEstimator
from arin_privacy_publication.estimator.student_t_test import StudentTTest


def compute_privacy_dmr(
    run_count: int,  # k
    dataset: DataFrame,  # d
    reference_rate: float,  # r
    distance: BaseDistance,  # d
    estimator: BaseEstimator,  # s
) -> float:
    """
    Compute the privacy DMR between two distributions.
    """
    if len(dataset) == 0:
        raise ValueError("d must not be empty")
    if len(dataset) % 2 != 0:
        raise ValueError("d must have an even number of elements")
    if reference_rate < 0 or 1 < reference_rate:
        raise ValueError("reference_rate must be in [0, 1]")
    if run_count <= 0:
        raise ValueError("run_count must be positive")

    n = len(dataset) // 2
    dr_size = int(reference_rate * n)
    if dr_size == 0:
        return 0.5
    succes_count = 0
    for _ in range(run_count):
        index = dataset.index.tolist()
        random.shuffle(index)
        d1 = dataset.iloc[index[:n]]
        d2 = dataset.iloc[index[n:]]
        dr = dataset.iloc[index[:dr_size]]  # because we shuffle d, we can just take the first dr_size elements
        if distance(estimator(d1), estimator(dr)) < distance(estimator(d2), estimator(dr)):
            succes_count += 1
    return succes_count / run_count


# def add_laplace_noise(sample: List[float], scale: float = 1.0, resolution: Optional[float] = None) -> List[float]:

#     sample = sample.copy()
#     # find std_dev to help us scale our noise addition
#     std_dev = np.std(sample)
#     # Generate our distribution
#     noise = np.random.laplace(0.0, 0.1 * std_dev * scale, 10000)
#     # For each row in our column
#     for i in range(len(sample)):
#         # scale noise to the resolution size for this column
#         if resolution is not None:
#             noise[i] = noise[i] - (noise[i] % resolution)
#         # Add the noise to the column
#         sample[i] += noise[i]
#     return sample


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
