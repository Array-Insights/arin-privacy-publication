from abc import ABC, abstractmethod
from typing import List

from arin_privacy_publication.base_noise import BaseNoise


class NoNoise(BaseNoise):
    def __init__(self):
        super().__init__(0)

    def __call__(self, sample: List[List[float]]) -> List[List[float]]:
        return sample
