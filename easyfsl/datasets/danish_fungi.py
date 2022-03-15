from pathlib import Path
from typing import List, Union, Tuple, Callable

import pandas as pd
from pandas import DataFrame
from PIL import Image
from torch import Tensor

from easyfsl.datasets import FewShotDataset
from easyfsl.datasets.default_configs import default_transform

WHOLE_DANISH_FUNGI_SPECS_FILE = Path("data/fungi") / "DF20_metadata.csv"


class DanishFungi(FewShotDataset):
    def __init__(
        self,
        root: Union[Path, str],
        specs_file: Union[Path, str] = WHOLE_DANISH_FUNGI_SPECS_FILE,
        image_size: int = 84,
        transform: Callable = None,
        training: bool = False,
    ):
        """
        Args:
            root: directory where all the images are
            specs_file: path to the CSV file
            image_size: images returned by the dataset will be square images of the given size
            transform: torchvision transforms to be applied to images. If none is provided,
                we use some standard transformations including ImageNet normalization.
                These default transformations depend on the "training" argument.
            training: preprocessing is slightly different for a training set, adding a random
                cropping and a random horizontal flip. Only used if transforms = None.
        """
        self.root = Path(root)
        self.data = self.load_specs(specs_file)

        self.class_names = list(self.data.drop_duplicates("label").scientific_name)

        self.transform = (
            transform if transform else default_transform(image_size, training=training)
        )

    @staticmethod
    def load_specs(specs_file: Path) -> DataFrame:
        """
        Load specs from a CSV file.
        Args:
            specs_file: path to the CSV file
        Returns:
            curated data contained in the CSV file
        """
        data = pd.read_csv(specs_file)

        class_names = list(data.scientific_name.unique())
        label_mapping = {name: class_names.index(name) for name in class_names}

        return data.assign(label=lambda df: df.scientific_name.map(label_mapping))

    def __getitem__(self, item: int) -> Tuple[Tensor, int]:
        """
        Get a data sample from its integer id.
        Args:
            item: sample's integer id
        Returns:
            data sample in the form of a tuple (image, label), where label is an integer.
            The type of the image object depends of the output type of self.transform.
        """
        img = self.transform(
            Image.open(self.root / self.data.image_path[item]).convert("RGB")
        )
        label = self.data.label[item]

        return img, label

    def __len__(self) -> int:
        return len(self.data)

    def get_labels(self) -> List[int]:
        return list(self.data.label)
