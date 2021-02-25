"""
General utilities
"""

from typing import List

import torchvision
from matplotlib import pyplot as plt
import numpy as np
import torch


def plot_images(images: torch.Tensor or List, title: str, images_per_row: int):
    """
    Plot images in a grid.
    Args:
        images: 4D mini-batch Tensor of shape (B x C x H x W) or a list of images of the same size
        title: title of the figure to plot
        images_per_row: number of images in each row of the grid
    """
    plt.figure()
    plt.title(title)
    plt.imshow(
        torchvision.utils.make_grid(images, nrow=images_per_row).permute(1, 2, 0)
    )


def sliding_average(value_list: List[float], window: int) -> float:
    """
    Computes the average of the latest instances in a list
    Args:
        value_list: input list of floats
        window: number of instances to take into account

    Returns:
        average of the last window instances in value_list
    """
    return np.asarray(value_list[-window:]).mean()
