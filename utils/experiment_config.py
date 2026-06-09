from pathlib import Path

import yaml

from config.config import get_config
from utils.paths import PROJECT_ROOT


EXPERIMENT_CONFIG_PATH = PROJECT_ROOT / "config" / "reproduce_experiments.yaml"

DATASET_ALIASES = {
    "handwritten": "handwritten",
    "Handwritten": "handwritten",
    "100leaves": "100leaves",
    "100Leaves": "100leaves",
    "100leaves": "100leaves",
    "Scene-15": "Scene-15",
    "Scene15": "Scene-15",
    "Scene-15_3V": "Scene-15",
    "LandUse-21": "LandUse-21",
    "LandUse": "LandUse-21",
    "LandUse-21_3V": "LandUse-21",
}

DATASET_FLAGS = {
    "handwritten": 0,
    "100leaves": 1,
    "Scene-15": 2,
    "LandUse-21": 3,
}

BOOST_KEY_MAP = {
    "refine_finetune_epoch": "refine_finetune_epochs",
    "sample_weight": "sample_weight_lambda",
    "w_vc": "w_vr",
    "w_cc": "w_cv",
    "w_rr": "w_rc",
    "early_stop_delta": "early_stop_min_delta",
}


def normalize_dataset_name(dataset):
    key = str(dataset).strip()
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    lower_key = key.lower()
    for alias, canonical in DATASET_ALIASES.items():
        if alias.lower() == lower_key:
            return canonical
    raise ValueError(f"Unsupported dataset in experiment config: {dataset}")


def dataset_to_flag(dataset):
    dataset_name = normalize_dataset_name(dataset)
    return DATASET_FLAGS[dataset_name]


def normalize_missing_rate(rate):
    return float(rate)


def missing_rates_equal(left, right, tol=1e-8):
    return abs(float(left) - float(right)) <= tol


def load_experiment_config(path=None):
    config_path = Path(path or EXPERIMENT_CONFIG_PATH)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    experiments = data.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError(f"{config_path} must contain an experiments list")
    return experiments


def find_experiment(dataset, missing_rate, path=None):
    dataset_name = normalize_dataset_name(dataset)
    target_rate = normalize_missing_rate(missing_rate)
    for experiment in load_experiment_config(path):
        exp_dataset = normalize_dataset_name(experiment.get("dataset"))
        exp_rate = normalize_missing_rate(experiment.get("missing_rate"))
        if exp_dataset == dataset_name and missing_rates_equal(exp_rate, target_rate):
            return experiment
    raise ValueError(
        f"No experiment config for dataset={dataset_name}, missing_rate={target_rate}"
    )


def build_config_from_experiment(dataset, missing_rate, seed=None, path=None):
    experiment = find_experiment(dataset, missing_rate, path)
    flag = dataset_to_flag(experiment["dataset"])
    config = get_config(flag)
    config["dataset"] = normalize_dataset_name(experiment["dataset"])
    config["missing_rate"] = normalize_missing_rate(experiment["missing_rate"])
    if seed is not None:
        config["seed"] = int(seed)
    return config


def get_experiment_seeds(dataset, missing_rate, path=None):
    experiment = find_experiment(dataset, missing_rate, path)
    seeds = experiment.get("seeds", [])
    if not seeds:
        raise ValueError(
            f"No seeds configured for dataset={dataset}, missing_rate={missing_rate}"
        )
    return [int(seed) for seed in seeds]


def normalize_boost_config(boost_config):
    normalized = {}
    for key, value in (boost_config or {}).items():
        mapped_key = BOOST_KEY_MAP.get(key, key)
        if mapped_key == "avg_select":
            continue
        normalized[mapped_key] = value
    return normalized


def get_boost_config(dataset, missing_rate, path=None):
    experiment = find_experiment(dataset, missing_rate, path)
    return normalize_boost_config(experiment.get("boost", {}))
