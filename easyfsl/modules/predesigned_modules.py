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
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet12(**kwargs):
    """Constructs a ResNet-12 model."""
    model = ResNet(BasicBlock, [1, 1, 2, 1], planes=[64, 160, 320, 640], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
