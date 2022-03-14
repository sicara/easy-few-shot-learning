from pathlib import Path
from typing import Callable, Optional, Union, List


import pandas as pd
from pandas import DataFrame
from PIL import Image
import torch
from tqdm import tqdm

from easyfsl.datasets import FewShotDataset
from easyfsl.datasets.default_configs import (
    default_mini_imagenet_loading_transform,
    default_mini_imagenet_serving_transform,
)

MINI_IMAGENET_SPECS_DIR = Path("data/mini_imagenet")


class MiniImageNet(FewShotDataset):
    def __init__(
        self,
        root: Union[Path, str],
        split: Optional[str] = None,
        specs_file: Optional[Union[Path, str]] = None,
        image_size: int = 84,
        loading_transform: Callable = None,
        transform: Callable = None,
        training: bool = False,
    ):
        self.root = Path(root)
        self.data_df = self.load_specs(split, specs_file)

        # Transformation to do before loading the dataset in RAM
        self.loading_transform = (
            loading_transform
            if loading_transform
            else default_mini_imagenet_loading_transform(image_size, training)
        )

        # Transformation to operate on the fly
        self.transform = (
            transform
            if transform
            else default_mini_imagenet_serving_transform(image_size, training)
        )

        self.images = torch.stack(
            [
                self.load_image_as_tensor(image_path)
                for image_path in tqdm(self.data_df.image_path)
            ]
        )

        self.class_names = self.data_df.class_name.unique()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):

        img, label = (
            self.transform(self.images[item]),
            self.labels[item],
        )

        return img, label

    def load_image_as_tensor(self, filename):
        return self.loading_transform(Image.open(filename).convert("RGB"))

    def load_specs(
        self,
        split: Optional[str] = None,
        specs_file: Optional[Union[Path, str]] = None,
    ) -> DataFrame:
        if (specs_file is None) & (split is None):
            raise ValueError("Please specify either a split or an explicit specs_file.")

        specs_file = (
            specs_file if specs_file else MINI_IMAGENET_SPECS_DIR / f"{split}.csv"
        )

        return pd.read_csv(specs_file).assign(
            image_path=lambda df: df.apply(
                lambda row: self.root / row["class_name"] / row["image_name"], axis=1
            )
        )

    def get_labels(self) -> List[int]:
        return list(self.data_df.class_name.map(self.class_to_id))
