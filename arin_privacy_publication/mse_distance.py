from typing import List

import numpy as np

from arin_privacy_publication.base_distance import BaseDistance


class MseDistance(BaseDistance):
    def __init__(self):
        pass

    def __call__(self, a: List[float], b: List[float]) -> float:
        return float(np.mean((np.array(a) - np.array(b)) ** 2))


# class safeMSE(BaseDistance):
#     def __init__(self):
#         self.epsilon = 0.5
#         pass

#     def __call__(self, a: List[float], b: List[float]) -> float:
#         sensitivity = self.calculate_dp_mse_sensitivity(a, b)
#         beta = sensitivity / self.epsilon

#         noisy_mse = np.mean((np.array(a) - np.array(b) + np.random.laplace(0, beta)) ** 2)

#         return noisy_mse

#     def calculate_mse(self, a, b):
#         # Calculate the mean squared error (MSE) between two lists a and b
#         mse = np.mean((np.array(a) - np.array(b) + np.random.laplace(0, beta)) ** 2)
#         return mse

#     def calculate_sensitivity(self, a, b, index):
#         # Calculate the sensitivity for a specific data point at index
#         mse_with = self.calculate_mse(a, b)
#         a_without = a[:index] + a[index + 1 :]
#         b_without = b[:index] + b[index + 1 :]
#         mse_without = self.calculate_mse(a_without, b_without)
#         sensitivity = abs(mse_with - mse_without)
#         return sensitivity

#     def calculate_dp_mse_sensitivity(self, a, b):
#         # Calculate the sensitivity of the differentially private MSE between lists a and b
#         num_data_points = len(a)
#         sensitivities = []

#         for i in range(num_data_points):
#             sensitivity = self.calculate_sensitivity(a, b, i)
#             sensitivities.append(sensitivity)

#         max_sensitivity = max(sensitivities)
#         return max_sensitivity
