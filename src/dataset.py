import json
from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

NORMALIZE_DEFAULT = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class EasySet(Dataset):
    def __init__(self, split_file: str, image_size=224, training=False):
        specs = json.load(open(split_file, "r"))

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
