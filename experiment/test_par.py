from math import sqrt

from joblib import Parallel, delayed
from tqdm import tqdm

result = Parallel(n_jobs=2)(delayed(sqrt)(i**2) for i in tqdm(range(100000)))
