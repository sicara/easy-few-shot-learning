from unittest.mock import patch

import pytest
import torch
from torch import nn
from torchvision.models import resnet18

from easyfsl.methods import AbstractMetaLearner


class TestAMLInit:
    @staticmethod
    @pytest.mark.parametrize(
        "backbone",
        [
            nn.Conv2d(3, 4, 4),
        ],
    )
    def test_constructor_raises_error_when_arg_is_not_a_feature_extractor(backbone):
        with pytest.raises(ValueError):
            AbstractMetaLearner(backbone)


# pylint: disable=not-callable
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
        with patch("torch.Tensor.cuda", new=torch.Tensor.cpu):
            with patch("easyfsl.methods.AbstractMetaLearner.forward") as mock_forward:
                with patch("easyfsl.methods.AbstractMetaLearner.process_support_set"):
                    mock_forward.return_value = torch.tensor(5 * [[0.25, 0.75]]).cuda()
                    model = AbstractMetaLearner(resnet18())
                    assert (
                        model.evaluate_on_one_task(
                            support_images,
                            support_labels,
                            query_images,
                            query_labels,
                        )
                        == (expected_correct, expected_total)
                    )


# pylint: enable=not-callable


class TestAMLAbstractMethods:
    @staticmethod
    def test_forward_raises_error_when_not_implemented():
        with pytest.raises(NotImplementedError):
            model = AbstractMetaLearner(resnet18())
            model(None)

    @staticmethod
    def test_process_support_set_raises_error_when_not_implemented():
        with pytest.raises(NotImplementedError):
            model = AbstractMetaLearner(resnet18())
            model.process_support_set(None, None)
