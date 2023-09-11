import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from easyfsl.methods.utils import compute_prototypes, entropy
from easyfsl.utils import (
    compute_average_features_from_images,
    plot_images,
    predict_embeddings,
    sliding_average,
)

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


class DummyDataset(Dataset):
    def __init__(self, images, class_names):
        self.images = images
        self.class_names = class_names

    def __getitem__(self, index):
        return self.images[index], self.class_names[index]

    def __len__(self):
        return len(self.class_names)


class TestPredictEmbeddings:
    cases_grid = [
        (
            DataLoader(
                DummyDataset(
                    torch.tensor(
                        [
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                        ],
                        dtype=torch.float64,
                    ),
                    ["class_1", "class_2", "class_2"],
                ),
                batch_size=3,
            ),
            pd.DataFrame(
                {
                    "embedding": [
                        torch.tensor(
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
                        ),
                        torch.tensor(
                            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float64
                        ),
                        torch.tensor(
                            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float64
                        ),
                    ],
                    "class_name": ["class_1", "class_2", "class_2"],
                }
            ),
        ),
        (
            DataLoader(
                DummyDataset(
                    torch.tensor(
                        [
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                        ],
                        dtype=torch.float64,
                    ),
                    ["class_1", "class_2", "class_2"],
                ),
                batch_size=2,
            ),
            pd.DataFrame(
                {
                    "embedding": [
                        torch.tensor(
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
                        ),
                        torch.tensor(
                            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float64
                        ),
                        torch.tensor(
                            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float64
                        ),
                    ],
                    "class_name": ["class_1", "class_2", "class_2"],
                }
            ),
        ),
        (
            DataLoader(
                DummyDataset(
                    torch.tensor(
                        [
                            [0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0],
                        ],
                        dtype=torch.float32,
                    ),
                    ["class_1", "class_2", "class_2"],
                ),
                batch_size=2,
            ),
            pd.DataFrame(
                {
                    "embedding": [
                        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                        torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
                        torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32),
                    ],
                    "class_name": ["class_1", "class_2", "class_2"],
                }
            ),
        ),
        (
            DataLoader(
                DummyDataset(
                    torch.tensor(
                        [
                            [0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0],
                        ],
                        dtype=torch.float32,
                    ),
                    [1, 2, 2],
                ),
                batch_size=2,
            ),
            pd.DataFrame(
                {
                    "embedding": [
                        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                        torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
                        torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32),
                    ],
                    "class_name": [1, 2, 2],
                }
            ),
        ),
    ]

    @staticmethod
    @pytest.mark.parametrize("dataloader, expected_dataframe", cases_grid)
    def test_predict_embeddings_returns_expected_dataframe(
        dataloader, expected_dataframe
    ):
        output_dataframe = predict_embeddings(dataloader, nn.Identity())
        pd.testing.assert_frame_equal(output_dataframe, expected_dataframe)


class TestComputeAverageFeaturesFromImages:
    cases_grid = [
        (
            DataLoader(
                DummyDataset(
                    torch.tensor(
                        [
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                        ],
                        dtype=torch.float64,
                    ),
                    ["class_1", "class_2", "class_2"],
                ),
                batch_size=3,
            ),
            torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float64),
        ),
        (
            DataLoader(
                DummyDataset(
                    torch.tensor(
                        [
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                        ],
                        dtype=torch.float64,
                    ),
                    ["class_1", "class_2", "class_2"],
                ),
                batch_size=2,
            ),
            torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float64),
        ),
        (
            DataLoader(
                DummyDataset(
                    torch.tensor(
                        [
                            [0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0],
                        ],
                        dtype=torch.float32,
                    ),
                    ["class_1", "class_2", "class_2"],
                ),
                batch_size=2,
            ),
            torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        ),
        (
            DataLoader(
                DummyDataset(
                    torch.tensor(
                        [
                            [0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0],
                        ],
                        dtype=torch.float32,
                    ),
                    [1, 2, 2],
                ),
                batch_size=2,
            ),
            torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        ),
    ]

    @staticmethod
    @pytest.mark.parametrize("dataloader, expected_average", cases_grid)
    def test_returns_expected_average(dataloader, expected_average):
        output_tensor = compute_average_features_from_images(dataloader, nn.Identity())
        torch.testing.assert_close(output_tensor, expected_average)
