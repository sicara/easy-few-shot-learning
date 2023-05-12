import numpy as np
import pandas as pd
import pytest
import torch.testing

from easyfsl.datasets import FeaturesDataset


def assert_dataset_equals(
    dataset, expected_embeddings, expected_labels, expected_class_names
):
    torch.testing.assert_close(dataset.embeddings, expected_embeddings)
    assert dataset.labels == expected_labels
    assert dataset.class_names == expected_class_names


class TestFromDataFrame:
    @staticmethod
    @pytest.mark.parametrize(
        "source_dataframe,expected_embeddings,expected_labels,expected_class_names",
        [
            (
                pd.DataFrame(
                    {
                        "embedding": [
                            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                            torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
                            torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32),
                        ],
                        "class_name": ["class_0", "class_1", "class_0"],
                    }
                ),
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0],
                    ],
                    dtype=torch.float32,
                ),
                [0, 1, 0],
                ["class_0", "class_1"],
            ),
            (
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
                        "class_name": ["class_0", "class_1", "class_0"],
                    }
                ),
                torch.tensor(
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                    ],
                    dtype=torch.float64,
                ),
                [0, 1, 0],
                ["class_0", "class_1"],
            ),
            (
                pd.DataFrame(
                    {
                        "embedding": [],
                        "class_name": [],
                    }
                ),
                torch.empty(0),
                [],
                [],
            ),
            (
                pd.DataFrame(
                    {
                        "embedding": [
                            np.array([0.0, 0.0, 0.0], dtype=np.float64),
                            np.array([1.0, 1.0, 1.0], dtype=np.float64),
                            np.array([2.0, 2.0, 2.0], dtype=np.float64),
                        ],
                        "class_name": ["class_0", "class_1", "class_0"],
                    }
                ),
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0],
                    ],
                    dtype=torch.float64,
                ),
                [0, 1, 0],
                ["class_0", "class_1"],
            ),
            (
                pd.DataFrame(
                    {
                        "embedding": [
                            np.array([0.0, 0.0, 0.0], dtype=np.float32),
                            np.array([1.0, 1.0, 1.0], dtype=np.float32),
                            np.array([2.0, 2.0, 2.0], dtype=np.float32),
                        ],
                        "class_name": ["class_0", "class_1", "class_0"],
                    }
                ),
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0],
                    ],
                    dtype=torch.float32,
                ),
                [0, 1, 0],
                ["class_0", "class_1"],
            ),
            (
                pd.DataFrame(
                    {
                        "embedding": [
                            np.array(
                                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
                            ),
                            np.array(
                                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64
                            ),
                            np.array(
                                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=np.float64
                            ),
                        ],
                        "class_name": ["class_0", "class_1", "class_0"],
                    }
                ),
                torch.tensor(
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                    ],
                    dtype=torch.float64,
                ),
                [0, 1, 0],
                ["class_0", "class_1"],
            ),
            (
                pd.DataFrame(
                    {
                        "embedding": [],
                        "class_name": [],
                    }
                ),
                torch.empty(0),
                [],
                [],
            ),
        ],
    )
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_init_from_dataframe_gives_expected_dataset(
        source_dataframe, expected_embeddings, expected_labels, expected_class_names
    ):
        dataset = FeaturesDataset.from_dataframe(source_dataframe)
        assert_dataset_equals(
            dataset, expected_embeddings, expected_labels, expected_class_names
        )

    @staticmethod
    @pytest.mark.parametrize(
        "source_dataframe",
        [
            pd.DataFrame(
                {
                    "embeddings": [
                        torch.tensor([0.0, 0.0, 0.0]),
                        torch.tensor([1.0, 1.0, 1.0]),
                        torch.tensor([2.0, 2.0, 2.0]),
                    ],
                    "class_name": ["class_0", "class_1", "class_0"],
                }
            ),
            pd.DataFrame(
                {
                    "embedding": [
                        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
                        torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
                    ],
                    "class_names": ["class_0", "class_1", "class_0"],
                }
            ),
            pd.DataFrame(
                {
                    "embeddings": [],
                    "class_names": [],
                }
            ),
            pd.DataFrame(
                {
                    "embedding": [
                        np.array([0.0, 0.0, 0.0]),
                        np.array([1.0, 1.0, 1.0]),
                        np.array([2.0, 2.0, 2.0]),
                    ],
                }
            ),
            pd.DataFrame(
                {
                    "class_name": ["class_0", "class_1", "class_0"],
                }
            ),
        ],
    )
    def test_init_from_dataframe_raises_error_on_invalid_df(source_dataframe):
        with pytest.raises(ValueError):
            FeaturesDataset.from_dataframe(source_dataframe)


class TestFromDict:
    dict_cases = [
        (
            {
                "class_0": torch.tensor(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
                ),
                "class_1": torch.tensor(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32
                ),
                "class_2": torch.tensor(
                    [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float32
                ),
            },
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                ],
                dtype=torch.float32,
            ),
            [0, 0, 1, 1, 2, 2],
            ["class_0", "class_1", "class_2"],
        ),
        (
            {
                "class_0": torch.tensor(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
                ),
                "class_1": torch.tensor(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float64
                ),
                "class_2": torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float64),
            },
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                ],
                dtype=torch.float64,
            ),
            [0, 0, 1, 1, 2],
            ["class_0", "class_1", "class_2"],
        ),
        (
            {
                "class_0": np.array(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
                ),
                "class_1": np.array(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64
                ),
                "class_2": np.array(
                    [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=np.float64
                ),
            },
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                ],
                dtype=torch.float64,
            ),
            [0, 0, 1, 1, 2, 2],
            ["class_0", "class_1", "class_2"],
        ),
        (
            {
                "class_0": np.array(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
                ),
                "class_1": np.array(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32
                ),
                "class_2": np.array(
                    [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=np.float32
                ),
            },
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                ],
                dtype=torch.float32,
            ),
            [0, 0, 1, 1, 2, 2],
            ["class_0", "class_1", "class_2"],
        ),
        (
            {
                "class_0": np.array(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
                ),
                "class_1": np.array(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32
                ),
                "class_2": torch.tensor(
                    [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float32
                ),
            },
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                ],
                dtype=torch.float32,
            ),
            [0, 0, 1, 1, 2, 2],
            ["class_0", "class_1", "class_2"],
        ),
        (
            {
                "class_0": torch.tensor(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
                ),
                "class_1": np.array(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32
                ),
                "class_2": torch.tensor(
                    [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float32
                ),
            },
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0],
                ],
                dtype=torch.float32,
            ),
            [0, 0, 1, 1, 2, 2],
            ["class_0", "class_1", "class_2"],
        ),
    ]

    @staticmethod
    @pytest.mark.parametrize(
        "source_dict,expected_embeddings,expected_labels,expected_class_names",
        dict_cases,
    )
    def test_init_from_dict_gives_expected_dataset(
        source_dict,
        expected_embeddings,
        expected_labels,
        expected_class_names,
    ):
        dataset = FeaturesDataset.from_dict(source_dict)
        assert_dataset_equals(
            dataset, expected_embeddings, expected_labels, expected_class_names
        )
