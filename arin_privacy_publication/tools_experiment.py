import json
import os
import random
from hashlib import sha256
from typing import Any, Callable, List, Optional

import numpy as np
from joblib import Parallel, delayed
from pandas import DataFrame
from tqdm import tqdm

from arin_privacy_publication.distance.base_distance import BaseDistance
from arin_privacy_publication.distribution.base_distribution import BaseDistribution
from arin_privacy_publication.estimator.base_estimator import BaseEstimator
from arin_privacy_publication.tools_privacy import compute_power, compute_privacy_dmr


def sha256_dict(dict: dict) -> str:
    return sha256(json.dumps(dict).encode()).hexdigest()


def run_experiment(dict_experiment: dict, ignore_cache: bool = False) -> dict:
    if not os.path.isdir("experiment_cache"):
        os.mkdir("experiment_cache")
    experiment_hash = sha256_dict(dict_experiment)
    path_file_experiment = os.path.join("experiment_cache", experiment_hash + ".json")
    if os.path.exists(path_file_experiment) and not ignore_cache:
        with open(path_file_experiment, "r") as file:
            return json.load(file)
    else:
        experiment_type = dict_experiment["experiment_type"]
        if experiment_type == "power":
            result = run_experiment_power(dict_experiment)
        elif experiment_type == "dmr":
            result = run_experiment_dmr(dict_experiment)
        elif experiment_type == "sensitivity":
            result = run_experiment_sensitivity(dict_experiment)
        else:
            raise ValueError(f"Unknown experiment type {experiment_type}")
        with open(path_file_experiment, "w") as file:
            json.dump(result, file)
        return result


def create_experiment_power(
    test: BaseEstimator,
    data_generator: BaseDistribution,
    sample_size: int,
    signigicance: float,
    run_count: int,
) -> dict:
    experiment = {}
    experiment["experiment_type"] = "power"
    experiment["test"] = test.to_dict()
    experiment["data_generator"] = data_generator.to_dict()
    experiment["sample_size"] = sample_size
    experiment["signigicance"] = signigicance
    experiment["run_count"] = run_count

    return experiment


def run_experiment_power(experiment: dict) -> dict:
    test: BaseEstimator = BaseEstimator.from_dict(experiment["test"])
    data_generator: BaseDistribution = BaseDistribution.from_dict(experiment["data_generator"])
    sample_size: int = experiment["sample_size"]
    signigicance: float = experiment["signigicance"]
    run_count: int = experiment["run_count"]

    power_mean = compute_power(test, data_generator, sample_size, signigicance, run_count)
    result = experiment.copy()
    result["result"] = {}
    result["result"]["power_mean"] = power_mean
    return result


def create_experiment_dmr(
    data_generator: BaseDistribution,
    sample_size: int,
    run_count: int,
    distance: BaseDistance,
    estimator: BaseEstimator,
    list_reference_rate: List[float],
) -> dict:
    experiment = {}
    experiment["experiment_type"] = "dmr"
    experiment["data_generator"] = data_generator.to_dict()
    experiment["sample_size"] = sample_size
    experiment["run_count"] = run_count
    experiment["distance"] = distance.to_dict()
    experiment["estimator"] = estimator.to_dict()
    experiment["list_reference_rate"] = list_reference_rate
    return experiment


def run_experiment_dmr(experiment: dict) -> dict:
    data_generator: BaseDistribution = BaseDistribution.from_dict(experiment["data_generator"])
    sample_size: int = experiment["sample_size"]
    run_count: int = experiment["run_count"]
    distance: BaseDistance = BaseDistance.from_dict(experiment["distance"])
    estimator: BaseEstimator = BaseEstimator.from_dict(experiment["estimator"])
    list_reference_rate: List[float] = experiment["list_reference_rate"]

    dataset = data_generator.sample(sample_size)
    list_dmr = []
    for reference_rate in list_reference_rate:
        dmr = compute_privacy_dmr(
            run_count,
            dataset,
            reference_rate,
            distance,
            estimator,
        )
        list_dmr.append(dmr)

    dmr_auc = np.trapz(list_dmr, list_reference_rate)
    result = experiment.copy()
    result["result"] = {}
    result["result"]["list_dmr"] = list_dmr
    result["result"]["dmr_auc"] = dmr_auc
    return result


def create_experiment_sensitivity(
    data_generator: BaseDistribution,
    sample_size: int,
    run_count: int,
    estimator: BaseEstimator,
) -> dict:
    experiment = {}
    experiment["experiment_type"] = "sensitivity"
    experiment["data_generator"] = data_generator.to_dict()
    experiment["sample_size"] = sample_size
    experiment["run_count"] = run_count
    experiment["estimator"] = estimator.to_dict()
    return experiment


def run_experiment_sensitivity(experiment: dict) -> dict:
    data_generator: BaseDistribution = BaseDistribution.from_dict(experiment["data_generator"])
    sample_size: int = experiment["sample_size"]
    run_count: int = experiment["run_count"]
    estimator: BaseEstimator = BaseEstimator.from_dict(experiment["estimator"])

    dataset = data_generator.sample(sample_size)
    list_sensitivity = []
    for _ in range(run_count):
        list_sensitivity.append(estimator.sensitivity(dataset))

    result = experiment.copy()
    result["result"] = {}
    result["result"]["sensitivity_sdev"] = float(np.std(list_sensitivity))
    result["result"]["sensitivity_mean"] = float(np.mean(list_sensitivity))
    return result


# def parfor_run_experiment(list_item: List[Any]) -> None:
#     parfor_tqdm(list_item, run_experiment)


# def parfor_tqdm(list_item: List[Any], function: Callable[[Any], Any]) -> None:
#     (delayed(function)(list_item[i]) for i in tqdm(range(len(list_item)))):
#         Parallel(n_jobs=2)
