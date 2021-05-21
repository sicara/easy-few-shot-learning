import torch
from torchvision.models import mobilenet_v2

from easyfsl.methods import MatchingNetworks


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
        model = MatchingNetworks(mobilenet_v2())

        model.process_support_set(support_images, support_labels)
        predictions = model(query_images)

        # pylint: disable=not-callable
        assert torch.all(
            torch.isclose(
                predictions,
                torch.tensor([[-0.3475, -1.2256], [-0.1823, -1.7919]]),
                rtol=1e-03,
            )
        )
        # pylint: enable=not-callable
