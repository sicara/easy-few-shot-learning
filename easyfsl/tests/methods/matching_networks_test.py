import pytest
import torch
from torch import nn

from easyfsl.methods import MatchingNetworks


class TestMatchingNetworksInit:
    @staticmethod
    @pytest.mark.parametrize(
        "backbone",
        [
            nn.Conv2d(3, 4, 4),
        ],
    )
    def test_constructor_raises_error_when_arg_is_not_a_feature_extractor(backbone):
        with pytest.raises(ValueError):
            MatchingNetworks(backbone)


class TestMatchingNetworksPipeline:
    @staticmethod
    def test_matching_networks_returns_expected_output_for_example_images(
        example_few_shot_classification_task,
    ):
        (
            support_images,
            support_labels,
            query_images,
        ) = example_few_shot_classification_task

        torch.manual_seed(1)
        torch.set_num_threads(1)

        model = MatchingNetworks(nn.Flatten())

        model.process_support_set(support_images, support_labels)
        predictions = model(query_images)

        # pylint: disable=not-callable
        assert torch.all(
            torch.isclose(
                predictions,
                torch.tensor([[-1.3137, -0.3131], [-1.0779, -0.4160]]),
                atol=1e-01,
            )
        )
        # pylint: enable=not-callable
