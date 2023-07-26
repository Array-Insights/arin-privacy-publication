from math import sqrt

import scipy
from joblib import Parallel, delayed
from tqdm import tqdm

print(scipy.stats.ttest_ind([0, 1, 2], [1, 2, 3], equal_var=True))

result = Parallel(n_jobs=2)(delayed(sqrt)(i**2) for i in tqdm(range(100000)))
