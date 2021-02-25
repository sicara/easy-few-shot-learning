import json
from pathlib import Path

import pandas as pd
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
        specs = json.load(open(specs_file, "r"))

        self.images = []
        self.labels = []
        for class_id, class_root in enumerate(specs["class_roots"]):
            class_images = [
                str(image_path)
                for image_path in Path(class_root).glob("*")
                if image_path.is_file()
            ]
            self.images += class_images
            self.labels += len(class_images) * [class_id]

        self.data = pd.concat(
            [
                pd.DataFrame(
                    {
                        "images": [
                            str(image_path)
                            for image_path in Path(class_root).glob("*")
                            if image_path.is_file()
                        ],
                        "labels": class_id,
                    }
                )
                for class_id, class_root in enumerate(specs["class_roots"])
            ],
            ignore_index=True,
        )

        self.class_names = specs["class_names"]

        self.transform = (
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

    def __getitem__(self, item):
        img = self.transform(Image.open(self.images[item]))
        label = self.labels[item]

        return img, label

    def __len__(self):
        return len(self.labels)
