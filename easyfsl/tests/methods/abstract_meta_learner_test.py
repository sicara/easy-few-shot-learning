from unittest.mock import patch

import pytest
import torch
from torch import nn
from torchvision.models import resnet18

from easyfsl.methods import AbstractMetaLearner


class TestAMLEvaluateOnOneTask:
    @staticmethod
    @pytest.mark.parametrize(
        "support_images,support_labels,query_images,query_labels,expected_correct,expected_total",
        [
            (
                torch.ones((5, 3, 28, 28)),
                torch.tensor([0, 0, 0, 0, 1]),
                torch.ones((5, 3, 28, 28)),
                torch.tensor([0, 0, 0, 0, 1]),
                1,
                5,
            ),
        ],
    )
    def test_evaluate_on_one_task_gives_correct_output(
        support_images,
        support_labels,
        query_images,
        query_labels,
        expected_correct,
        expected_total,
    ):
        with patch("easyfsl.methods.AbstractMetaLearner.forward") as mock_forward:
            mock_forward.return_value = torch.tensor(5 * [[0.25, 0.75]]).cuda()
            model = AbstractMetaLearner(nn.Conv2d(1, 1, 1))
            assert (
                model.evaluate_on_one_task(
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                )
                == (expected_correct, expected_total)
            )
