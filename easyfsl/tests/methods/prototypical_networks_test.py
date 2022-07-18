import pytest
import torch
from torch import nn

from easyfsl.datasets import SupportSetFolder
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


class TestProtoNetsCanProcessSupportSetFolder:
    @staticmethod
    @pytest.mark.parametrize(
        "support_set_path",
        [
            "easyfsl/tests/datasets/resources/balanced_support_set",
            "easyfsl/tests/datasets/resources/unbalanced_support_set",
        ],
    )
    def test_proto_nets_can_process_support_set_from_balanced_folder(
        support_set_path, dummy_network
    ):
        support_set = SupportSetFolder(support_set_path)
        support_images = support_set.get_images()
        support_labels = support_set.get_labels()

        model = PrototypicalNetworks(backbone=dummy_network)
        model.process_support_set(support_images, support_labels)

        query_images = torch.randn((4, 3, 224, 224))
        model(query_images)

    @staticmethod
    @pytest.mark.parametrize(
        (
            "support_set_path",
            "expected_prototypes",
        ),
        [
            (
                "easyfsl/tests/datasets/resources/unbalanced_support_set",
                [-0.0987, -0.0489, -0.3414],
            ),
            (
                "easyfsl/tests/datasets/resources/balanced_support_set",
                [-0.0987, 0.2805, -0.3582],
            ),
        ],
    )
    def test_proto_nets_store_correct_prototypes(
        support_set_path, expected_prototypes, deterministic_dummy_network
    ):
        support_set = SupportSetFolder(support_set_path)
        support_images = support_set.get_images()
        support_labels = support_set.get_labels()

        model = PrototypicalNetworks(backbone=deterministic_dummy_network)
        model.process_support_set(support_images, support_labels)

        assert torch.all(
            torch.isclose(
                model.prototypes,
                torch.tensor(expected_prototypes).unsqueeze(1),
                atol=1e-04,
            )
        )
