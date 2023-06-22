import random
from typing import List, Optional

import numpy as np

from arin_privacy_publication.base_distance import BaseDistance
from arin_privacy_publication.base_noise import BaseNoise
from arin_privacy_publication.base_statistic import BaseStatistic
from arin_privacy_publication.no_noise import NoNoise


def compute_privacy_dmr(
    k: int,
    d: List[float],
    r: float,
    f: BaseDistance,
    s: BaseStatistic,
    noise: Optional[BaseNoise] = None,
) -> float:
    """
    Compute the privacy DMR between two distributions.
    """
    if len(d) == 0:
        raise ValueError("d must not be empty")
    if len(d) % 2 != 0:
        raise ValueError("d must have an even number of elements")
    if r < 0 or r > 1:
        raise ValueError("r must be in [0, 1]")
    if k <= 0:
        raise ValueError("k must be positive")

    if noise is None:
        noise = NoNoise()

    n = len(d) // 2
    dr_size = int(r * n)
    if dr_size == 0:
        return 0.5
    succes_count = 0
    for _ in range(k):
        random.shuffle(d)
        d1 = d[:n]
        d2 = d[n:]
        dr = d1[:dr_size]  # because we shuffle d, we can just take the first dr_size elements
        if f(s(noise(d1)), s(dr)) < f(s(noise(d2)), s(dr)):
            succes_count += 1
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
