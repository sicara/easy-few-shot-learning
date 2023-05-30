from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from easyfsl.datasets import (
    CUB,
    DanishFungi,
    FewShotDataset,
    MiniImageNet,
    TieredImageNet,
)
from easyfsl.methods import (
    BDCSPN,
    FEAT,
    PTMAP,
    TIM,
    Finetune,
    LaplacianShot,
    MatchingNetworks,
    PrototypicalNetworks,
    RelationNetworks,
    SimpleShot,
    TransductiveFinetuning,
)
from easyfsl.modules.build_from_checkpoint import feat_resnet12_from_checkpoint
from easyfsl.utils import predict_embeddings

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

BACKBONES_DICT = {
    "feat_resnet12": feat_resnet12_from_checkpoint,
}

DATASETS_DICT = {
    "CUB": CUB,
    "tieredImageNet": TieredImageNet,
    "miniImageNet": MiniImageNet,
    "fungi": DanishFungi,
}
DEFAULT_FUNGI_PATH = Path("data/fungi/images")


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
        num_workers: The number of workers to use. Defaults to 0 for no multiprocessing.
        output_parquet: Where to save the extracted embeddings. Defaults to
            {backbone}_{dataset}_{split}.parquet.gzip in the current directory.
    """
    model = build_backbone(backbone, checkpoint, device)

    initialized_dataset = get_dataset(dataset, split)
    dataloader = DataLoader(
        initialized_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    embeddings_df = predict_embeddings(dataloader, model, device=device)

    if output_parquet is None:
        output_parquet = Path(f"{backbone}_{dataset}_{split}.parquet.gzip")
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    embeddings_df.to_parquet(output_parquet, index=False, compression="gzip")
    logger.info(f"Saved embeddings to {output_parquet}")


def get_dataset(dataset_name: str, split: str) -> FewShotDataset:
    if dataset_name not in DATASETS_DICT:
        raise ValueError(
            "Unknown dataset name." f"Valid names are {DATASETS_DICT.keys()}"
        )
    if dataset_name == "fungi":
        if split != "test":
            raise ValueError("Danish Fungi only has a test set.")
        return DanishFungi(DEFAULT_FUNGI_PATH)
    return DATASETS_DICT[dataset_name](split=split, training=False)


def build_backbone(
    backbone: str,
    checkpoint: Path,
    device: str,
) -> nn.Module:
    if backbone not in BACKBONES_DICT:
        raise ValueError(
            "Unknown backbone name." f"Valid names are {BACKBONES_DICT.keys()}"
        )
    backbone = BACKBONES_DICT[backbone](checkpoint, device)
    backbone.eval()

    return backbone


if __name__ == "__main__":
    typer.run(main)
