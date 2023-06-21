import random
from typing import List

from arin_privacy_publication.base_distance import BaseDistance
from arin_privacy_publication.base_statistic import BaseStatistic


def compute_privacy_dmr(
    k: int,
    d: List[float],
    r: float,
    f: BaseDistance,
    s: BaseStatistic,
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
        if f(s(d1), s(dr)) < f(s(d2), s(dr)):
            succes_count += 1
    return succes_count / k
