import pytest
import torch
from torchvision.models.resnet import BasicBlock

from easyfsl.modules import ResNet


class TestResNetForward:
    @staticmethod
    @pytest.mark.parametrize(
        "layers,planes,output_size",
        [
            (
                [1, 1, 1, 1],
                [16, 32, 14, 8],
                8,
            ),
            (
                [1, 1, 3, 4],
                [16, 32, 14, 8],
                8,
            ),
            (
                [1, 1, 1, 1],
                [16, 32, 14, 1],
                1,
            ),
            (
                [1, 1, 1, 1],
                [16, 32, 14, 8],
                8,
            ),
            (
                [1, 1, 1, 1],
                [4, 4, 4, 4],
                4,
            ),
        ],
    )
    def test_basicblock_resnets_output_vector_of_correct_size_without_fc(
        layers, planes, output_size
    ):
        n_images = 5

        model = ResNet(
            block=BasicBlock,
            layers=layers,
            planes=planes,
            use_fc=False,
            use_pooling=True,
        )

        input_images = torch.ones((n_images, 3, 84, 84))

        assert model(input_images).shape == (n_images, output_size)
