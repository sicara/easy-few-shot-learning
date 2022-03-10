from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


@pytest.fixture
def example_few_shot_classification_task():
    images_dir = Path("easyfsl/tests/methods/resources")
    support_image_paths = [
        "Black_footed_Albatross_0001_2950163169.jpg",
        "Black_footed_Albatross_0002_2293084168.jpg",
        "Least_Auklet_0001_2947317867.jpg",
    ]
    query_image_paths = [
        "Black_footed_Albatross_0004_2731401028.jpg",
        "Least_Auklet_0004_2685272855.jpg",
    ]
    support_labels = torch.tensor([0, 0, 1])  # pylint: disable=not-callable

    to_tensor = transforms.ToTensor()
    support_images = torch.stack(
        [
            to_tensor(Image.open(images_dir / img_name))
            for img_name in support_image_paths
        ]
    )
    query_images = torch.stack(
        [to_tensor(Image.open(images_dir / img_name)) for img_name in query_image_paths]
    )

    return support_images, support_labels, query_images


@pytest.fixture()
def dummy_network():
    return nn.Sequential(
        nn.Flatten(),
        nn.AdaptiveAvgPool1d(output_size=10),
        nn.Linear(10, 5),
    )
