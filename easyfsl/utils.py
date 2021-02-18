from typing import List

import torchvision
from matplotlib import pyplot as plt
import numpy as np
import torch


def plot_images(images: torch.Tensor, title: str, images_per_row: int):
    plt.figure()
    plt.title(title)
    plt.imshow(
        torchvision.utils.make_grid(images, nrow=images_per_row).permute(1, 2, 0)
    )


def sliding_average(value_list: List[float], window: int) -> float:
    return np.asarray(value_list[-window:]).mean()
