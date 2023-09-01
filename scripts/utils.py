import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
GRID_SEARCH_JSON = Path("scripts/grid_search.json")


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


def build_model(
    method: str,
    device: str,
    **kwargs,
) -> FewShotClassifier:
    """
    Build a model from a method name and an optional config.
    Args:
        method: must be one of the keys of METHODS_DICT
        device: device to cast the model to (cpu or cuda)
        kwargs: optional hyperparameters to initialize the model
    Returns:
        the requested FewShotClassifier
    """
    if method not in METHODS_DICT:
        raise ValueError(
            "Unknown method name. " f"Valid names are {METHODS_DICT.keys()}"
        )

    if method == "feat":
        return FEAT.from_resnet12_checkpoint(
            **kwargs, device=device, use_backbone=False
        )

    model = METHODS_DICT[method](nn.Identity(), **kwargs).to(device)
    return model


def get_dataloader_from_features_path(
    features: Path,
    n_way: int,
    n_shot: int,
    n_query: int,
    n_tasks: int,
    num_workers: int,
):
    """
    Build a dataloader from a path to a pickle file containing a dict mapping labels to all their embeddings.
    Args:
        features: path to a Parquet or Pickle file containing the features.
        n_way: number of classes per task.
        n_shot: number of support example per class.
        n_query: number of query instances per class.
        n_tasks: number of tasks to evaluate on.
        num_workers: The number of workers to use for the DataLoader.
    Returns:
        a DataLoader that yields features in the shape of a task
    """
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
    return features_loader


def get_dataset(features_path: Path) -> FeaturesDataset:
    """
    Load a FeaturesDataset from a path to either a pickle file containing a dict mapping labels to all their embeddings,
    or a parquet file containing a dataframe with the columns "embedding" and "label".
    Args:
        features_path: path to a pickle or parquet file containing the features.
    Returns:
        a FeaturesDataset
    """
    if features_path.suffix == ".pickle":
        embeddings_dict = pd.read_pickle(features_path)
        return FeaturesDataset.from_dict(embeddings_dict)
    embeddings_df = pd.read_parquet(features_path)
    return FeaturesDataset.from_dataframe(embeddings_df)
