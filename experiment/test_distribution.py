from pandas import DataFrame

from arin_privacy_publication.distribution.exponential import Exponential
from arin_privacy_publication.distribution.laplace import Laplace
from arin_privacy_publication.distribution.normal import Normal
from arin_privacy_publication.distribution.uniform import Uniform

list_data_generator = [
    Normal([0, 1], [1, 2]),
    Laplace([0, 1], [1, 2]),
    Uniform([0, 1], [1, 2]),
    Exponential([0, 1], [1, 2]),
]

for data_generator in list_data_generator:
    print(data_generator.distribution_name)
    data_frame: DataFrame = data_generator.sample(10000)
    print(data_frame["series_0"].mean())
    print(data_frame["series_0"].std())

    print(data_frame["series_1"].mean())
    print(data_frame["series_1"].std())

# Lévy distribution
# np.random.exponential(standard_deviation, count) + shift).tolist()
