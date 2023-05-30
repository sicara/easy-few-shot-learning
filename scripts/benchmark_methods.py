import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import typer
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from easyfsl.datasets import FeaturesDataset
from easyfsl.methods import (
    BDCSPN,
    FEAT,
    PTMAP,
    TIM,
    FewShotClassifier,
    Finetune,
    LaplacianShot,
    MatchingNetworks,
    PrototypicalNetworks,
    RelationNetworks,
    SimpleShot,
    TransductiveFinetuning,
)
from easyfsl.samplers import TaskSampler
from easyfsl.utils import evaluate

METHODS_DICT = {
    "bd_cspn": BDCSPN,
    "feat": FEAT,
    "finetune": Finetune,
    "laplacian_shot": LaplacianShot,
    "matching_networks": MatchingNetworks,
    "prototypical_networks": PrototypicalNetworks,
    "pt_map": PTMAP,
    "relation_networks": RelationNetworks,
    "simple_shot": SimpleShot,
    "tim": TIM,
    "transductive_finetuning": TransductiveFinetuning,
}
METHODS_CONFIGS_JSON = Path("scripts/methods_configs.json")


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
    Evaluate a method on a dataset of features pre-extracted by a backbone.
    Args:
        method: Few-Shot Classifier to use.
        features: path to a Parquet file containing the features.
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

    model = build_model(method, device, config)
    logger.info(f"Loaded model {method} with {config} config.")

    features_dataset = get_dataset(features)
    task_sampler = TaskSampler(
        features_dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_tasks,
    )
    features_loader = DataLoader(
        features_dataset,
        batch_sampler=task_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=task_sampler.episodic_collate_fn,
    )
    logger.info(f"Loaded features from {features}")

    accuracy = evaluate(model, features_loader, device)
    logger.info(f"Average accuracy : {(100 * accuracy):.2f} %")


def set_random_seed(seed: int):
    """
    Set random, numpy and torch random seed, for reproducibility of the training
    Args:
        seed: defined random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(features_path: Path) -> FeaturesDataset:
    embeddings_df = pd.read_parquet(features_path)
    return FeaturesDataset.from_dataframe(embeddings_df)


def build_model(
    method: str,
    device: str,
    config: Optional[str] = None,
) -> FewShotClassifier:
    """
    Build a model from a method name and an optional config.
    Args:
        method: must be one of the keys of METHODS_DICT
        device: device to cast the model to (cpu or cuda)
        config: the name of a config for the specified method in scripts/methods_configs.json
    Returns:
        the requested FewShotClassifier
    """
    if method not in METHODS_DICT:
        raise ValueError(
            "Unknown method name. " f"Valid names are {METHODS_DICT.keys()}"
        )
    if config is not None:
        retrieved_config = read_config(method, config)
    else:
        retrieved_config = {}

    if method == "feat":
        return FEAT.from_resnet12_checkpoint(
            **retrieved_config, device=device, use_backbone=False
        )

    model = METHODS_DICT[method](nn.Identity(), **retrieved_config).to(device)
    return model


def read_config(method: str, config: str) -> dict:
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
