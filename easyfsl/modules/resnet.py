from typing import Type, Union, List

import torch.nn as nn
import torch
from torch import Tensor
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1

# pylint: disable=invalid-name, too-many-instance-attributes, too-many-arguments


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        planes: List[int] = None,
        use_fc: bool = False,
        num_classes: int = 1000,
        use_pooling: bool = True,
        big_kernel: bool = False,
        zero_init_residual: bool = False,
    ):
        """
        Custom ResNet architecture, with some design differences compared to the built-in
        PyTorch ResNet.
        This implementation and its usage in predesigned_modules is derived from
        https://github.com/fiveai/on-episodes-fsl/blob/master/src/models/ResNet.py
        Args:
            block: which core block to use (BasicBlock, Bottleneck, or any child of one of these)
            layers: number of blocks in each of the 4 layers
            planes: number of planes in each of the 4 layers
            use_fc: whether to use one last linear layer on features
            num_classes: output dimension of the last linear layer (only used if use_fc is True)
            use_pooling: whether to average pool the features (must be True if use_fc is True)
            big_kernel: whether to use the shape of the built-in PyTorch ResNet designed for
                ImageNet. If False, make the first convolutional layer less destructive.
            zero_init_residual: zero-initialize the last BN in each residual branch, so that the
                residual branch starts with zeros, and each residual block behaves like an identity.
                This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        """
        super().__init__()
        if planes is None:
            planes = [64, 128, 256, 512]

        self.inplanes = 64

        # Built-in PyTorch ResNet uses a first conv layer with a 7*7 kernel and a stride of 2,
        # which is fine for ImageNet's 224x224 images, but too destructive for 84x84 images
        # which are commonly used in few-shot settings.
        self.conv1 = (
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=1, bias=False)
            if big_kernel
            else nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, planes[0], layers[0])
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2)

        self.use_pooling = use_pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Only used when self.use_fc is True
        self.use_fc = use_fc
        self.fc = nn.Linear(self.inplanes, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3.weight, 0)
                elif isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer4(
            self.layer3(self.layer2(self.layer1(self.relu(self.bn1(self.conv1(x))))))
        )

        if self.use_pooling:
            x = torch.flatten(
                self.avgpool(x),
                1,
            )

            if self.use_fc:
                return self.fc(x)

        else:
            if self.use_fc:
                raise ValueError(
                    "You can't use the fully connected layer without pooling features."
                )

        return x

    def set_use_fc(self, use_fc: bool):
        """
        Change the use_fc property. Allow to decide when and where the model should use its last
        fully connected layer.
        Args:
            use_fc: whether to set self.use_fc to True or False
        """
        self.use_fc = use_fc


# pylint: enable=invalid-name, too-many-instance-attributes, too-many-arguments
