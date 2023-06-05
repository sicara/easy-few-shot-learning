import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from easyfsl.datasets import (
    CUB,
    DanishFungi,
    FewShotDataset,
    MiniImageNet,
    TieredImageNet,
)
from easyfsl.modules.build_from_checkpoint import feat_resnet12_from_checkpoint
from easyfsl.utils import predict_embeddings

BACKBONES_DICT = {
    "feat_resnet12": feat_resnet12_from_checkpoint,
}
BACKBONES_CONFIGS_JSON = Path("scripts/backbones_configs.json")
INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}

DATASETS_DICT = {
    "cub": CUB,
    "fungi": DanishFungi,
    "mini_imagenet": MiniImageNet,
    "tiered_imagenet": TieredImageNet,
}
DEFAULT_FUNGI_PATH = Path("data/fungi/images")
DEFAULT_MINI_IMAGENET_PATH = Path("data/mini_imagenet/images")


def main(
    backbone: str,
    checkpoint: Path,
    dataset: str,
    split: str = "test",
    device: str = "cuda",
    batch_size: int = 128,
    num_workers: int = 0,
    output_parquet: Optional[Path] = None,
) -> None:
    """
    Use a pretrained backbone to extract embeddings from a dataset, and save them as Parquet.
    Args:
        backbone: The name of the backbone to use.
        checkpoint: The path to the checkpoint to use.
        dataset: The name of the dataset to use.
        split: Which split to use among train, val test. Some datasets only have a test split.
        device: The device to use.
        batch_size: The batch size to use.
        num_workers: The number of workers to use for the DataLoader. Defaults to 0 for no multiprocessing.
        output_parquet: Where to save the extracted embeddings. Defaults to
            {backbone}_{dataset}_{split}.parquet.gzip in the current directory.
    """
    model = build_backbone(backbone, checkpoint, device)
    logger.info(f"Loaded backbone {backbone} from {checkpoint}")

    dataset_transform = get_dataset_transform(backbone)

    initialized_dataset = get_dataset(dataset, split, dataset_transform)
    dataloader = DataLoader(
        initialized_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    logger.info(f"Loaded dataset {dataset} ({split} split)")

    embeddings_df = predict_embeddings(dataloader, model, device=device)
    cast_embeddings_to_numpy(embeddings_df)

    if output_parquet is None:
        output_parquet = (
            Path("data/features")
            / dataset
            / split
            / checkpoint.with_suffix(".parquet.gzip").name
        )
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    embeddings_df.to_parquet(output_parquet, index=False, compression="gzip")
    logger.info(f"Saved embeddings to {output_parquet}")


def build_backbone(
    backbone: str,
    checkpoint: Path,
    device: str,
) -> nn.Module:
    """
    Build a backbone from a checkpoint.
    Args:
        backbone: name of the backbone. Must be a key of BACKBONES_DICT.
        checkpoint: path to the checkpoint
        device: device on which to build the backbone
    Returns:
        The backbone, loaded from the checkpoint, and in eval mode.
    """
    if backbone not in BACKBONES_DICT:
        raise ValueError(
            "Unknown backbone name. " f"Valid names are {BACKBONES_DICT.keys()}"
        )
    model = BACKBONES_DICT[backbone](checkpoint, device)
    model.eval()

    return model


def get_dataset_transform(backbone_name: str) -> transforms.Compose:
    """
    Get the transform to apply to the images before feeding them to the backbone.
    Use the config defined for the specified backbone at scripts/backbones_configs.json.
    Args:
        backbone_name: must be a key in scripts/backbones_configs.json.
    Returns:
        A callable to apply to the images, with a resize, a center-crop, a conversion to tensor, and a normalization.
    """
    with open(BACKBONES_CONFIGS_JSON, "r", encoding="utf-8") as file:
        all_configs = json.load(file)
    if backbone_name not in all_configs:
        raise ValueError(
            f"No available config for {backbone_name} in {str(BACKBONES_CONFIGS_JSON)}."
        )
    transform_config = all_configs[backbone_name]["transform"]
    return transforms.Compose(
        [
            transforms.Resize(
                int(transform_config["image_size"] * transform_config["crop_ratio"]),
                interpolation=INTERPOLATIONS[transform_config["interpolation"]],
            ),
            transforms.CenterCrop(transform_config["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(transform_config["mean"], transform_config["std"]),
        ]
    )


def get_dataset(
    dataset_name: str, split: str, transform: transforms.Compose
) -> FewShotDataset:
    """
    Get a dataset using the built-in constructors from EasyFSL.
    Args:
        dataset_name: must be one of "cub", "tiered_imagenet", "mini_imagenet", "fungi".
        split: train, val, or test
        transform: a callable to apply to the images.
    Returns:
        The requested dataset.
    """
    if dataset_name not in DATASETS_DICT:
        raise ValueError(
            "Unknown dataset name. " f"Valid names are {DATASETS_DICT.keys()}"
        )
    if dataset_name == "fungi":
        if split != "test":
            raise ValueError("Danish Fungi only has a test set.")
        return DanishFungi(DEFAULT_FUNGI_PATH, transform=transform)
    if dataset_name == "mini_imagenet":
        return MiniImageNet(
            root=DEFAULT_MINI_IMAGENET_PATH,
            split=split,
            training=False,
            transform=transform,
        )
    return DATASETS_DICT[dataset_name](split=split, training=False, transform=transform)


def cast_embeddings_to_numpy(embeddings_df: pd.DataFrame) -> None:
    """
    Cast the tensor embeddings in a DataFrame to numpy arrays, in an inplace fashion.
    Args:
        embeddings_df: dataframe with an "embeddings" column containing torch tensors.
    """
    embeddings_df["embedding"] = embeddings_df["embedding"].apply(
        lambda embedding: embedding.detach().cpu().numpy()
    )


if __name__ == "__main__":
    typer.run(main)
