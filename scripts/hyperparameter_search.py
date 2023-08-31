import json
from itertools import product
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from loguru import logger

from easyfsl.utils import evaluate
from scripts.utils import (
    GRID_SEARCH_JSON,
    build_model,
    get_dataloader_from_features_path,
    set_random_seed,
)


def main(  # pylint: disable=too-many-locals
    method: str,
    features: Path,
    n_way: int = 5,
    n_shot: int = 5,
    n_query: int = 15,
    n_tasks: int = 500,
    device: str = "cuda",
    num_workers: int = 0,
    random_seed: int = 0,
    output_csv: Optional[Path] = None,
) -> None:
    """
    Perform hyperparameter grid search for a method on a dataset of features pre-extracted by a backbone.
    Outputs the results in a csv file with all tested combinations of parameters and the corresponding accuracy.
    Args:
        method: Few-Shot Classifier to use.
        features: path to a Parquet or Pickle file containing the features.
        n_way: number of classes per task.
        n_shot: number of support example per class.
        n_query: number of query instances per class.
        n_tasks: number of tasks to evaluate on.
        device: device to use
        num_workers: The number of workers to use for the DataLoader. Defaults to 0 for no multiprocessing.
        random_seed: random seed to use for reproducibility.
        output_csv: path to the output csv file.
    """
    set_random_seed(random_seed)
    hyperparameter_grid_df = read_hyperparameter_grid(method)
    logger.info(
        f"Loaded {len(hyperparameter_grid_df)} hyperparameter combinations for {method}."
    )

    features_loader = get_dataloader_from_features_path(
        features, n_way, n_shot, n_query, n_tasks, num_workers
    )
    logger.info(f"Loaded features from {features}")

    accuracies_record = []
    for config_dict in iter(hyperparameter_grid_df.to_dict(orient="records")):
        model = build_model(method, device, **config_dict)
        logger.info(f"Loaded model {method} with following config:")
        logger.info(json.dumps(config_dict, indent=4))
        accuracy = evaluate(model, features_loader, device)
        accuracies_record.append(accuracy)
        logger.info(f"Average accuracy : {(100 * accuracy):.2f} %")

    hyperparameter_grid_df = hyperparameter_grid_df.assign(accuracy=accuracies_record)
    logger.info(f"Hyperparameter search results for {method}:")
    logger.info("Best hyperparameters:")
    logger.info(
        json.dumps(
            hyperparameter_grid_df.sort_values("accuracy", ascending=False)
            .iloc[0]
            .to_dict(),
            indent=4,
        )
    )

    if output_csv is None:
        output_csv = Path(f"{method}_hyperparameter_search.csv")

    hyperparameter_grid_df.to_csv(output_csv, index=False)
    logger.info(f"Saved results in {output_csv}")


def read_hyperparameter_grid(method: str) -> pd.DataFrame:
    with open(GRID_SEARCH_JSON, "r", encoding="utf-8") as file:
        all_grids = json.load(file)
    if method not in all_grids:
        raise ValueError(
            f"No available hyperparameter grid for {method} in {str(GRID_SEARCH_JSON)}."
        )
    grid = all_grids[method]
    return pd.DataFrame(unroll_grid(grid))


def unroll_grid(input_dict: dict[str, list]) -> list[dict]:
    """
    Unroll a grid of hyperparameters into a list of dicts.
    Args:
        input_dict: each key is a parameter name, each value is a list of values for this parameter.
    Returns:
        a list of dicts, each dict is a combination of parameters.
    Examples:
        >>> unroll_grid({"a": [1, 2], "b": [3, 4]})
        [{"a": 1, "b": 3}, {"a": 1, "b": 4}, {"a": 2, "b": 3}, {"a": 2, "b": 4}]
    """
    return [
        dict(zip(input_dict.keys(), values)) for values in product(*input_dict.values())
    ]


if __name__ == "__main__":
    typer.run(main)
