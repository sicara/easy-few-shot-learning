import pytest
import torch
from torch import nn

from easyfsl.methods import RelationNetworks


class TestPrototypicalNetworksInit:
    @staticmethod
    @pytest.mark.parametrize(
        "backbone",
        [
            nn.Flatten(),
        ],
    )
    def test_constructor_raises_error_when_arg_doesnt_output_3d_feature_maps(backbone):
        with pytest.raises(ValueError):
            RelationNetworks(backbone)


class TestRelationNetworksPipeline:
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

        model = RelationNetworks(nn.Identity())

        model.relation_module = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten()
        )

        model.process_support_set(support_images, support_labels)
        predictions = model(query_images)

        # pylint: disable=not-callable
        assert torch.all(
            torch.isclose(
                predictions,
                torch.tensor(
                    [[0.4148, 0.4866], [0.6354, 0.7073]],
                ),
                rtol=1e-3,
            ),
        )
        # pylint: enable=not-callable
