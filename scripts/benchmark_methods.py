import json
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from easyfsl.utils import evaluate
from scripts.utils import (
    METHODS_CONFIGS_JSON,
    build_model,
    get_dataloader_from_features_path,
    set_random_seed,
)


def main(
    method: str,
    features: Path,
    config: Optional[str] = None,
    n_way: int = 5,
    n_shot: int = 5,
    n_query: int = 15,
    n_tasks: int = 1000,
    device: str = "cuda",
    num_workers: int = 0,
    random_seed: int = 0,
) -> None:
    """
    Evaluate a method on a dataset of features pre-extracted by a backbone. Print the average accuracy.
    Args:
        method: Few-Shot Classifier to use.
        features: path to a Parquet or Pickle file containing the features.
        config: existing configuration for the method available in scripts/methods_configs.json
        n_way: number of classes per task.
        n_shot: number of support example per class.
        n_query: number of query instances per class.
        n_tasks: number of tasks to evaluate on.
        device: device to use
        num_workers: The number of workers to use for the DataLoader. Defaults to 0 for no multiprocessing.
        random_seed: random seed to use for reproducibility.
    """
    set_random_seed(random_seed)

    config_dict = read_config(method, config)
    model = build_model(method, device, **config_dict)
    logger.info(f"Loaded model {method} with {config} config.")

    features_loader = get_dataloader_from_features_path(
        features, n_way, n_shot, n_query, n_tasks, num_workers
    )
    logger.info(f"Loaded features from {features}")

    accuracy = evaluate(model, features_loader, device)
    logger.info(f"Average accuracy : {(100 * accuracy):.2f} %")


def read_config(method: str, config: Optional[str]) -> dict:
    if config is None:
        return {}
    with open(METHODS_CONFIGS_JSON, "r", encoding="utf-8") as file:
        all_configs = json.load(file)
    if method not in all_configs:
        raise ValueError(
            f"No available config for {method} in {str(METHODS_CONFIGS_JSON)}."
        )
    configs = all_configs[method]
    if config not in configs:
        raise ValueError(
            f"No available config {config} for {method} in {str(METHODS_CONFIGS_JSON)}."
        )
    return configs[config]


if __name__ == "__main__":
    typer.run(main)
