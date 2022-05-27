from pathlib import Path
from typing import Callable, Optional, Union, List


import pandas as pd
from pandas import DataFrame
from PIL import Image
import torch
from torch import Tensor
from tqdm import tqdm

from easyfsl.datasets import FewShotDataset
from easyfsl.datasets.default_configs import (
    default_mini_imagenet_loading_transform,
    default_mini_imagenet_serving_transform,
    default_transform,
)

MINI_IMAGENET_SPECS_DIR = Path("data/mini_imagenet")


class MiniImageNet(FewShotDataset):
    def __init__(
        self,
        root: Union[Path, str],
        split: Optional[str] = None,
        specs_file: Optional[Union[Path, str]] = None,
        image_size: int = 84,
        load_on_ram: bool = False,
        loading_transform: Callable = None,
        transform: Callable = None,
        training: bool = False,
    ):
        """
        Build the miniImageNet dataset from specific specs file. By default all images are loaded
        in RAM at construction time. Otherwise images are loaded on the fly.
        Args:
            root: directory where all the images are
            split: if specs_file is not specified, will look for the CSV file corresponding
                to this split in miniImageNet's specs directory. If both are unspecified,
                raise an error.
            specs_file: path to the specs CSV file. Mutually exclusive with split but one of them
                must be specified.
            image_size: images returned by the dataset will be square images of the given size
            load_on_ram: if True, images are processed through loading_transform then stored on RAM.
                If False , images are loaded on the fly. Preloading demands available space on RAM
                and a few minutes at construction time, but will save a lot of time during training.
            loading_transform: only used if load_on_ram is True. Torchvision transforms to be
                applied to images during preloading. Must contain ToTensor. If none is provided, we
                use standard transformations (Resize if training is False, RandomResizedCrop
                if True)
            transform: torchvision transforms to be applied to images. If none is provided,
                we use some standard transformations including ImageNet normalization.
                These default transformations depend on the "training" argument.
                If load_on_ram is False, default transformations include default loading
                transformations.
            training: preprocessing is slightly different for a training set, adding a random
                cropping and a random horizontal flip. Only used if transforms = None.
        """
        self.root = Path(root)
        self.data_df = self.load_specs(split, specs_file)
        self.load_on_ram = load_on_ram

        if self.load_on_ram:
            # Transformation to do before loading the dataset in RAM
            self.loading_transform = (
                loading_transform
                if loading_transform
                else default_mini_imagenet_loading_transform(image_size)
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
                    for image_path in tqdm(
                        self.data_df.image_path, desc="Loading images"
                    )
                ]
            )

        else:
            self.loading_transform = None
            self.transform = (
                transform if transform else default_transform(image_size, training)
            )
            self.images = self.data_df.image_path.tolist()

        self.class_names = self.data_df.class_name.unique()
        self.class_to_label = {v: k for k, v in enumerate(self.class_names)}
        self.labels = self.get_labels()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, item):
        img = (
            self.transform(self.images[item])
            if self.load_on_ram
            else self.transform(
                Image.open(self.data_df.image_path[item]).convert("RGB")
            )
        )

        return img, self.labels[item]

    def load_image_as_tensor(self, filename) -> Tensor:
        return self.loading_transform(Image.open(filename).convert("RGB"))

    def load_specs(
        self,
        split: Optional[str] = None,
        specs_file: Optional[Union[Path, str]] = None,
    ) -> DataFrame:
        """
        Load the classes and paths of images from the CSV specs file.
        Args:
            split: if specs_file is not specified, will look for the CSV file corresponding
                to this split in miniImageNet's specs directory. If both are unspecified,
                raise an error.
            specs_file: path to the specs CSV file. Mutually exclusive with split but one of them
                must be specified.

        Returns:
            dataframe with 3 columns class_name, image_name and image_path

        Raises:
            ValueError: you need to specify a split or a specs_file, but not both.
        """
        if (specs_file is None) & (split is None):
            raise ValueError("Please specify either a split or an explicit specs_file.")
        if (specs_file is not None) & (split is not None):
            raise ValueError("Conflict: you can't specify a split AND a specs file.")

        specs_file = (
            specs_file if specs_file else MINI_IMAGENET_SPECS_DIR / f"{split}.csv"
        )

        return pd.read_csv(specs_file).assign(
            image_path=lambda df: df.apply(
                lambda row: self.root / row["class_name"] / row["image_name"], axis=1
            )
        )

    def get_labels(self) -> List[int]:
        return list(self.data_df.class_name.map(self.class_to_label))
