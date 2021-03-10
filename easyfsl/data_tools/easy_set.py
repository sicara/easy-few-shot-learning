import json
from pathlib import Path
from typing import List

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


NORMALIZE_DEFAULT = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class EasySet(Dataset):
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

    def __init__(self, specs_file: Path, image_size=224, training=False):
        """
        Args:
            specs_file: path to the JSON file
            image_size: images returned by the dataset will be square images of the given size
            training: preprocessing is slightly different for a training set, adding a random
                cropping and a random horizontal flip.
        """
        specs = self.load_specs(specs_file)

        self.images, self.labels = self.list_data_instances(specs["class_roots"])

        self.class_names = specs["class_names"]

        self.transform = self.compose_transforms(image_size, training)

    @staticmethod
    def load_specs(specs_file: Path) -> dict:
        """
        Load specs from a JSON file.
        Args:
            specs_file: path to the JSON file

        Returns:
            dictionary contained in the JSON file
        """

        if specs_file.suffix != ".json":
            raise ValueError("EasySet requires specs in a JSON file.")

        specs = json.load(open(specs_file, "r"))

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
    def compose_transforms(image_size: int, training: bool) -> transforms.Compose:
        """
        Create a composition of torchvision transformations, with some randomization if we are
            building a training set.
        Args:
            image_size: size of dataset images
            training: whether this is a training set or not

        Returns:
            compositions of torchvision transformations
        """
        return (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(**NORMALIZE_DEFAULT),
                ]
            )
            if training
            else transforms.Compose(
                [
                    transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(**NORMALIZE_DEFAULT),
                ]
            )
        )

    @staticmethod
    def list_data_instances(class_roots: List[str]) -> (List[str], List[int]):
        """
        Explore the directories specified in class_roots to find all data instances.
        Args:
            class_roots: each element is the path to the directory containing the elements
                of one class

        Returns:
            list of paths to the images, and a list of same length containing the integer label
                of each image
        """
        images = []
        labels = []
        for class_id, class_root in enumerate(class_roots):
            class_images = [
                str(image_path)
                for image_path in Path(class_root).glob("*")
                if image_path.is_file()
            ]
            images += class_images
            labels += len(class_images) * [class_id]

        return images, labels

    def __getitem__(self, item):
        img = self.transform(Image.open(self.images[item]))
        label = self.labels[item]

        return img, label

    def __len__(self):
        return len(self.labels)
