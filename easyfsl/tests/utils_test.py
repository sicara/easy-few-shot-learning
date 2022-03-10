import pytest
import torch
from torchvision import transforms

from easyfsl.utils import plot_images, sliding_average, compute_prototypes, entropy

TO_PIL_IMAGE = transforms.ToPILImage()


class TestPlotImages:
    @staticmethod
    @pytest.mark.parametrize(
        "images,title,images_per_row",
        [
            (torch.ones((10, 3, 5, 5)), "", 5),
            (torch.ones((10, 3, 5, 5)), "title", 15),
        ],
    )
    def test_function_does_not_break(images, title, images_per_row):
        plot_images(images, title, images_per_row)


class TestSlidingAverage:
    @staticmethod
    @pytest.mark.parametrize(
        "value_list,window,expected_mean",
        [
            ([0.1, 0.0, 1.0], 2, 0.5),
            ([0.1, 0.0, 1], 2, 0.5),
            ([0.1, 0.0, 1.0], 1, 1.0),
            ([0.0, 0.5, 1.0], 3, 0.5),
            ([0.0, 0.5, 1.0], 4, 0.5),
            ([0.0, 0.5, 1.0], 0, 0.5),
        ],
    )
    def test_returns_correct_mean(value_list, window, expected_mean):
        assert sliding_average(value_list, window) == expected_mean

    @staticmethod
    @pytest.mark.parametrize(
        "value_list,window",
        [
            ([], 2),
            ([], 0),
        ],
    )
    def test_refuses_illegal_values(value_list, window):
        with pytest.raises(ValueError):
            sliding_average(value_list, window)


# pylint: disable=not-callable
class TestComputePrototypes:
    @staticmethod
    @pytest.mark.parametrize(
        "features, labels, expected_prototypes",
        [
            (
                torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
                torch.tensor([0, 0]),
                torch.tensor([[0.5, 0.5]]),
            ),
            (
                torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
                torch.tensor([0, 1]),
                torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            ),
            (
                torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
                torch.tensor([1, 0]),
                torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
            ),
        ],
    )
    def test_compute_prototypes_returns_correct_prototypes(
        features, labels, expected_prototypes
    ):
        assert torch.equal(compute_prototypes(features, labels), expected_prototypes)


# pylint: enable=not-callable


class TestEntropy:
    @staticmethod
    @pytest.mark.parametrize(
        "input_",
        [
            torch.ones((5, 4)),
            torch.ones((5, 1)),
            torch.ones((1, 5)),
        ],
    )
    def test_entropy_returns_correctly_shaped_tensor(input_):
        assert entropy(input_).shape == ()
