from typing import List

import numpy as np

from arin_privacy_publication.base_noise import BaseNoise


class LaplaceNoise(BaseNoise):
    def __init__(self, std_multiplier: float):
        super().__init__(std_multiplier)

    def __call__(self, sample: List[float]) -> List[float]:
        noise = np.random.laplace(0.0, self.std_multiplier / np.sqrt(2), len(sample))
        return (np.array(sample) + noise).tolist()
