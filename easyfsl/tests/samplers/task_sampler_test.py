from typing import Tuple, List

import pytest
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
            "labels": ["label0", "label0", "label0", 1, 1, 1, 2, 2, 2],
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
