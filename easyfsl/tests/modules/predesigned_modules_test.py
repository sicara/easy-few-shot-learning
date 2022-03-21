import pytest
import torch

from easyfsl.modules import (
    resnet10,
    resnet12,
    resnet18,
    resnet34,
    resnet50,
)


class TestResNets:
    all_resnets = [
        resnet10,
        resnet12,
        resnet18,
        resnet34,
        resnet50,
    ]

    @staticmethod
    @pytest.mark.parametrize("network", all_resnets)
    def test_resnets_instantiate_without_error(network):
        network()

    @staticmethod
    @pytest.mark.parametrize("network", all_resnets)
    def test_resnets_output_vector_of_size_num_classes_with_use_fc(network):
        num_classes = 10
        n_images = 5

        model = network(use_fc=True, num_classes=num_classes)

        input_images = torch.ones((n_images, 3, 84, 84))

        assert model(input_images).shape == (n_images, num_classes)
