import pytest
import torch
from torch import nn

from easyfsl.methods import PrototypicalNetworks


class TestPrototypicalNetworksInit:
    @staticmethod
    @pytest.mark.parametrize(
        "backbone",
        [
            nn.Conv2d(3, 4, 4),
        ],
    )
    def test_constructor_raises_error_when_arg_is_not_a_feature_extractor(backbone):
        with pytest.raises(ValueError):
            PrototypicalNetworks(backbone)


class TestPrototypicalNetworksPipeline:
    @staticmethod
    def test_prototypical_networks_returns_expected_output_for_example_images(
        example_few_shot_classification_task,
    ):
        (
            support_images,
            support_labels,
            query_images,
        ) = example_few_shot_classification_task

        torch.manual_seed(1)
        torch.set_num_threads(1)

        model = PrototypicalNetworks(nn.Flatten())

        model.process_support_set(support_images, support_labels)
        predictions = model(query_images)

        # pylint: disable=not-callable
        assert torch.all(
            torch.isclose(
                predictions,
                torch.tensor(
                    [[-15.5485, -22.0652], [-21.3081, -18.0292]],
                ),
                atol=1e-01,
            )
        )
        # pylint: enable=not-callable
