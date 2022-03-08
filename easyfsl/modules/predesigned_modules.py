from torchvision.models.resnet import BasicBlock, Bottleneck

from easyfsl.modules import ResNet

__all__ = [
    "resnet10",
    "resnet12",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


def resnet10(**kwargs):
    """Constructs a ResNet-10 model."""
    return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)


def resnet12(**kwargs):
    """Constructs a ResNet-12 model."""
    return ResNet(BasicBlock, [1, 1, 2, 1], planes=[64, 160, 320, 640], **kwargs)


def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    """Constructs a ResNet-152 model."""
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
