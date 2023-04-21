from typing import List, Tuple

import numpy as np
import pytest
import torch
from torch import Tensor

from easyfsl.datasets import FewShotDataset
from easyfsl.samplers import TaskSampler


class DummyFewShotDataset(FewShotDataset):
    def __init__(self, labels):
        self.labels = labels

    def __getitem__(self, item: int) -> Tuple[Tensor, int]:
        raise NotImplementedError("__getitem__() is not supposed to be called.")

    def __len__(self) -> int:
        raise NotImplementedError("__len__() is not supposed to be called.")

    def get_labels(self) -> List[int]:
        return self.labels


def init_task_sampler(labels, n_way, n_shot, n_query, n_tasks):
    return TaskSampler(
        dataset=DummyFewShotDataset(labels=labels),
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_tasks,
    )


class TestTaskSamplerIter:
    cases_grid = [
        {
            "labels": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "n_way": 2,
            "n_shot": 1,
            "n_query": 1,
            "n_tasks": 5,
        },
        {
            "labels": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "n_way": 2,
            "n_shot": 1,
            "n_query": 1,
            "n_tasks": 5,
        },
        {
            "labels": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "n_way": 3,
            "n_shot": 2,
            "n_query": 1,
            "n_tasks": 5,
        },
    ]

    @staticmethod
    @pytest.mark.parametrize(
        "labels,n_way,n_shot,n_query,n_tasks",
        [tuple(case.values()) for case in cases_grid],
    )
    def test_task_sampler_iter_yields_list_of_int(
        labels, n_way, n_shot, n_query, n_tasks
    ):
        sampler = init_task_sampler(labels, n_way, n_shot, n_query, n_tasks)
        for batch in sampler:
            assert isinstance(batch, list)
            for item in batch:
                assert isinstance(item, int)

    @staticmethod
    @pytest.mark.parametrize(
        "labels,n_way,n_shot,n_query,n_tasks",
        [tuple(case.values()) for case in cases_grid],
    )
    def test_task_sampler_iter_yields_list_of_correct_len(
        labels, n_way, n_shot, n_query, n_tasks
    ):
        sampler = init_task_sampler(labels, n_way, n_shot, n_query, n_tasks)
        for batch in sampler:
            assert len(batch) == n_way * (n_shot + n_query)

    @staticmethod
    @pytest.mark.parametrize(
        "labels,n_way,n_shot,n_query,n_tasks",
        [tuple(case.values()) for case in cases_grid],
    )
    def test_task_sampler_iter_yields_items_smaller_than_dataset_len(
        labels, n_way, n_shot, n_query, n_tasks
    ):
        sampler = init_task_sampler(labels, n_way, n_shot, n_query, n_tasks)
        for batch in sampler:
            for item in batch:
                assert item < len(labels)


class TestTaskSamplerSizeChecks:
    @staticmethod
    @pytest.mark.parametrize(
        "labels,n_way",
        [
            (
                [0, 0, 1, 1, 2, 2],
                4,
            ),
            (
                [0, 0],
                2,
            ),
        ],
    )
    def test_init_raises_error_when_number_of_classes_is_smaller_than_n_way(
        labels, n_way
    ):
        with pytest.raises(ValueError):
            init_task_sampler(labels, n_way, 1, 1, 1)

    @staticmethod
    @pytest.mark.parametrize(
        "labels,n_way",
        [
            (
                [0, 0, 1, 1, 2, 2],
                2,
            ),
            (
                [0, 0, 1, 1, 2, 2],
                3,
            ),
            (
                [0, 0],
                1,
            ),
        ],
    )
    def test_init_raises_no_error_when_number_of_classes_is_not_smaller_than_n_way(
        labels, n_way
    ):
        init_task_sampler(labels, n_way, 1, 1, 1)

    @staticmethod
    @pytest.mark.parametrize(
        "labels,n_shot,n_query",
        [
            (
                [0, 0, 1, 1, 2, 2],
                1,
                2,
            ),
            (
                [0, 0, 0, 1, 1, 1, 2, 2],
                3,
                0,
            ),
            (
                [0, 0],
                2,
                1,
            ),
        ],
    )
    def test_init_raises_error_when_a_label_population_is_too_small(
        labels, n_shot, n_query
    ):
        with pytest.raises(ValueError):
            init_task_sampler(labels, 1, n_shot, n_query, 1)

    @staticmethod
    @pytest.mark.parametrize(
        "labels,n_shot,n_query",
        [
            (
                [0, 0, 1, 1, 2, 2],
                1,
                1,
            ),
            (
                [0, 0, 0, 1, 1, 1, 2, 2],
                2,
                0,
            ),
            (
                [0, 0],
                0,
                1,
            ),
        ],
    )
    def test_init_raises_no_error_when_a_label_population_is_ok(
        labels, n_shot, n_query
    ):
        init_task_sampler(labels, 1, n_shot, n_query, 1)


class TestCastInputDataToTensorIntTuple:
    @staticmethod
    @pytest.mark.parametrize(
        "input_data,expected_error",
        [
            (
                [
                    (np.ones((2, 2)), 0),
                    (np.ones((2, 2)), 0),
                    (np.ones((2, 2)), 1),
                    (np.ones((2, 2)), 1),
                ],
                TypeError,
            ),
            (
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 1),
                    (np.ones((2, 2)), 1),
                ],
                TypeError,
            ),
            (
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 0.5),
                    (torch.ones((2, 2)), 1),
                    (torch.ones((2, 2)), 1),
                ],
                TypeError,
            ),
            (
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), "0"),
                    (torch.ones((2, 2)), 1),
                    (torch.ones((2, 2)), 1),
                ],
                TypeError,
            ),
            (
                [
                    (torch.ones((2, 2)), 0),
                    (np.ones((2, 2)), "0"),
                    (torch.ones((2, 2)), 1),
                    (torch.ones((2, 2)), 1),
                ],
                TypeError,
            ),
            (
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), "0"),
                    (np.ones((2, 2)), 1),
                    (torch.ones((2, 2)), 1),
                ],
                TypeError,
            ),
            (
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), "0"),
                    (torch.ones((2, 2)), torch.ones((2, 2), dtype=torch.int8)),
                    (torch.ones((2, 2)), 1),
                ],
                TypeError,
            ),
            (
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), torch.ones((2, 2), dtype=torch.int8)),
                    (torch.ones((2, 2)), "0"),
                    (torch.ones((2, 2)), 1),
                ],
                ValueError,
            ),
            (
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), torch.ones((2, 2), dtype=torch.int8)),
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 1),
                ],
                ValueError,
            ),
            (
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), torch.ones((2, 2), dtype=torch.int8)),
                    (np.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 1),
                ],
                ValueError,
            ),
            (
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), torch.ones((1, 2), dtype=torch.int8)),
                    (np.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 1),
                ],
                ValueError,
            ),
            (
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), torch.ones(2, dtype=torch.int8)),
                    (np.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 1),
                ],
                ValueError,
            ),
            (
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), torch.ones(1, dtype=torch.int8)),
                    (np.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 1),
                ],
                ValueError,
            ),
            (
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), torch.tensor(0.5)),
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 1),
                ],
                TypeError,
            ),
            (
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), torch.tensor(True)),
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 1),
                ],
                TypeError,
            ),
            (
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), torch.tensor(False)),
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 1),
                ],
                TypeError,
            ),
            (
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), torch.ones((2, 2), dtype=torch.float16)),
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 1),
                ],
                TypeError,
            ),
        ],
    )
    def test_raises_error_for_inappropriate_input(
        input_data, expected_error
    ):  # pylint: disable=protected-access
        sampler = init_task_sampler([0, 0, 1, 1, 2, 2], 2, 1, 1, 1)
        with pytest.raises(expected_error):
            sampler._cast_input_data_to_tensor_int_tuple(input_data)

    @staticmethod
    @pytest.mark.parametrize(
        "input_data,expected_output",
        [
            (
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 1),
                    (torch.ones((2, 2)), 1),
                ],
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 1),
                    (torch.ones((2, 2)), 1),
                ],
            ),
            (
                [
                    (torch.ones((2, 2)), torch.tensor(0)),
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), torch.tensor(1)),
                    (torch.ones((2, 2)), 1),
                ],
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 1),
                    (torch.ones((2, 2)), 1),
                ],
            ),
            (
                [
                    (torch.ones((2, 2)), torch.tensor(0)),
                    (torch.ones((2, 2)), torch.tensor(0)),
                    (torch.ones((2, 2)), torch.tensor(1)),
                    (torch.ones((2, 2)), torch.tensor(1)),
                ],
                [
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 0),
                    (torch.ones((2, 2)), 1),
                    (torch.ones((2, 2)), 1),
                ],
            ),
        ],
    )
    def test_correctly_casts_input_data(
        input_data, expected_output
    ):  # pylint: disable=protected-access
        sampler = init_task_sampler([0, 0, 1, 1, 2, 2], 2, 1, 1, 1)
        output = sampler._cast_input_data_to_tensor_int_tuple(input_data)
        assert len(output) == len(expected_output)
        for output_instance, expected_output_instance in zip(output, expected_output):
            assert output_instance[0].equal(expected_output_instance[0])
            assert output_instance[1] == expected_output_instance[1]
