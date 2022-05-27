import json
import warnings
from pathlib import Path
from typing import List, Union, Set, Tuple, Callable

from PIL import Image

from easyfsl.datasets import FewShotDataset
from easyfsl.datasets.default_configs import default_transform, DEFAULT_IMAGE_FORMATS


class EasySet(FewShotDataset):
    """
    A ready-to-use dataset. Will work for any dataset where the images are
    grouped in directories by class. It expects a JSON file defining the
    classes and where to find them. It must have the following shape:
        {
            "class_names": [
                "class_1",
                "class_2"
            ],
            "class_roots": [
                "path/to/class_1_folder",
                "path/to/class_2_folder"
            ]
        }
    """

    def __init__(
        self,
        specs_file: Union[Path, str],
        image_size: int = 84,
        transform: Callable = None,
        training: bool = False,
        supported_formats: Set[str] = None,
    ):
        """
        Args:
            specs_file: path to the JSON file
            image_size: images returned by the dataset will be square images of the given size
            transform: torchvision transforms to be applied to images. If none is provided,
                we use some standard transformations including ImageNet normalization.
                These default transformations depend on the "training" argument.
            training: preprocessing is slightly different for a training set, adding a random
                cropping and a random horizontal flip. Only used if transforms = None.
            supported_formats: set of allowed file format. When listing data instances, EasySet
                will only consider these files. If none is provided, we use the default set of
                image formats.
        """
        specs = self.load_specs(Path(specs_file))

        self.images, self.labels = self.list_data_instances(
            specs["class_roots"], supported_formats=supported_formats
        )

        self.class_names = specs["class_names"]

        self.transform = (
            transform if transform else default_transform(image_size, training)
        )

    @staticmethod
    def load_specs(specs_file: Path) -> dict:
        """
        Load specs from a JSON file.
        Args:
            specs_file: path to the JSON file

        Returns:
            dictionary contained in the JSON file

        Raises:
            ValueError: if specs_file is not a JSON, or if it is a JSON and the content is not
                of the expected shape.
        """

        if specs_file.suffix != ".json":
            raise ValueError("EasySet requires specs in a JSON file.")

        with open(specs_file, "r") as file:
            specs = json.load(file)

        if "class_names" not in specs.keys() or "class_roots" not in specs.keys():
            raise ValueError(
                "EasySet requires specs in a JSON file with the keys class_names and class_roots."
            )

        if len(specs["class_names"]) != len(specs["class_roots"]):
            raise ValueError(
                "Number of class names does not match the number of class root directories."
            )

        return specs

    @staticmethod
    def list_data_instances(
        class_roots: List[str], supported_formats: Set[str] = None
    ) -> Tuple[List[str], List[int]]:
        """
        Explore the directories specified in class_roots to find all data instances.
        Args:
            class_roots: each element is the path to the directory containing the elements
                of one class

        Returns:
            list of paths to the images, and a list of same length containing the integer label
                of each image
        """
        if supported_formats is None:
            supported_formats = DEFAULT_IMAGE_FORMATS

        images = []
        labels = []
        for class_id, class_root in enumerate(class_roots):
            class_images = [
                str(image_path)
                for image_path in sorted(Path(class_root).glob("*"))
                if image_path.is_file()
                & (image_path.suffix.lower() in supported_formats)
            ]

            images += class_images
            labels += len(class_images) * [class_id]

        if len(images) == 0:
            warnings.warn(
                UserWarning(
                    "No images found in the specified directories. The dataset will be empty"
                )
            )

        return images, labels

    def __getitem__(self, item: int):
        """
        Get a data sample from its integer id.
        Args:
            item: sample's integer id

        Returns:
            data sample in the form of a tuple (image, label), where label is an integer.
            The type of the image object depends of the output type of self.transform. By default
            it's a torch.Tensor, however you are free to define any function as self.transform, and
            therefore any type for the output image. For instance, if self.transform = lambda x: x,
            then the output image will be of type PIL.Image.Image.
        """
        # Some images of ILSVRC2015 are grayscale, so we convert everything to RGB for consistence.
        # If you want to work on grayscale images, use torch.transforms.Grayscale in your
        # transformation pipeline.
        img = self.transform(Image.open(self.images[item]).convert("RGB"))
        label = self.labels[item]

        return img, label

    def __len__(self) -> int:
        return len(self.labels)

    def get_labels(self) -> List[int]:
        return self.labels

    def number_of_classes(self):
        return len(self.class_names)
