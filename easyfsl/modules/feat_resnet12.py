"""
This particular ResNet12 is simplified from the original implementation of FEAT (https://github.com/Sha-Lab/FEAT).
We provide it to allow the reproduction of the FEAT method and the use of the chekcpoints they made available.
It contains some design choices that differ from the usual ResNet12. Use this one or the other.
Just remember that it is important to use the same backbone for a fair comparison between methods.
"""

from torch import nn
from torchvision.models.resnet import conv3x3


class FEATBasicBlock(nn.Module):
    """
    BasicBlock for FEAT. Uses 3 convolutions instead of 2, a LeakyReLU instead of ReLU, and a MaxPool2d.
    """

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample

    def forward(self, x):  # pylint: disable=invalid-name
        """
        Pass input through the block, including an activation and maxpooling at the end.
        """

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.relu(out)
        out = self.maxpool(out)

        return out


class FEATResNet12(nn.Module):
    """
    ResNet12 for FEAT. See feat_resnet12 doc for more details.
    """

    def __init__(
        self,
        block=FEATBasicBlock,
    ):
        self.inplanes = 3
        super().__init__()

        channels = [64, 160, 320, 640]
        self.layer_dims = [
            channels[i] * block.expansion for i in range(4) for j in range(4)
        ]

        self.layer1 = self._make_layer(
            block,
            64,
            stride=2,
        )
        self.layer2 = self._make_layer(
            block,
            160,
            stride=2,
        )
        self.layer3 = self._make_layer(
            block,
            320,
            stride=2,
        )
        self.layer4 = self._make_layer(
            block,
            640,
            stride=2,
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
            )
        )
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):  # pylint: disable=invalid-name
        """
        Iterate over the blocks and apply them sequentially.
        """
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return x.mean((-2, -1))


def feat_resnet12(**kwargs):
    """
    Build a ResNet12 model as used in the FEAT paper, following the implementation of
    https://github.com/Sha-Lab/FEAT.
    This ResNet network also follows the practice of the following papers:
    TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
    A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

    There are 4 main differences with the other ResNet models used in EasyFSL:
        - There is no first convolutional layer (3x3, 64) before the first block.
        - The stride of the first block is 2 instead of 1.
        - The BasicBlock uses 3 convolutional layers, instead of 2 in the standard torch implementation.
        - We don't initialize the last fully connected layer, since we never use it.

    Note that we removed the Dropout logic from the original implementation, as it is not part of the paper.

    Args:
        **kwargs: Additional arguments to pass to the FEATResNet12 class.

    Returns:
        The standard ResNet12 from FEAT model.
    """
    return FEATResNet12(FEATBasicBlock, **kwargs)
