from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
from pandas import DataFrame
from PIL import Image
from torch import Tensor

from .default_configs import default_transform
from .few_shot_dataset import FewShotDataset

WHOLE_DANISH_FUNGI_SPECS_FILE = Path("data/fungi") / "DF20_metadata.csv"


class DanishFungi(FewShotDataset):
    def __init__(
        self,
        root: Union[Path, str],
        specs_file: Union[Path, str] = WHOLE_DANISH_FUNGI_SPECS_FILE,
        image_size: int = 84,
        transform: Optional[Callable] = None,
        training: bool = False,
        image_file_extension: str = ".JPG",
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
            image_file_extension: the metadata csv file and the complete dataset user ".JPG" image file extension,
                but the version of the dataset with 300px images uses ".jpg" extensions. If using the small dataset,
                set this to ".jpg".
        """
        self.root = Path(root)
        self.image_file_extension = image_file_extension
        self.data = self.load_specs(Path(specs_file))

        self.class_names = list(self.data.drop_duplicates("label").scientific_name)

        self.transform = (
            transform if transform else default_transform(image_size, training=training)
        )

    def load_specs(self, specs_file: Path) -> DataFrame:
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

        if self.image_file_extension != ".JPG":
            data.image_path = data.image_path.str.replace(
                ".JPG", self.image_file_extension
            )

        return data.assign(label=lambda df: df.scientific_name.map(label_mapping))

    def __getitem__(self, item: int) -> Tuple[Tensor, int]:
        """
        Get a data sample from its integer id.
        Args:
            item: sample's integer id
        Returns:
            data sample in the form of a tuple (image, label), where label is an integer.
            The type of the image object depends on the output type of self.transform.
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
