from typing import Callable

from torchvision import transforms


IMAGENET_NORMALIZATION = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

DEFAULT_IMAGE_FORMATS = {".bmp", ".png", ".jpeg", ".jpg"}


def default_transform(image_size: int, training: bool) -> Callable:
    """
    Create a composition of torchvision transformations, with some randomization if we are
        building a training set.
    Args:
        image_size: size of dataset images
        training: whether this is a training set or not

    Returns:
        compositions of torchvision transformations
    """
    return (
        transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**IMAGENET_NORMALIZATION),
            ]
        )
        if training
        else transforms.Compose(
            [
                transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(**IMAGENET_NORMALIZATION),
            ]
        )
    )


def default_mini_imagenet_loading_transform(
    image_size: int,
) -> Callable:
    """
    Create a composition of torchvision transformations to perform when loading images, but before
    serving them (when data is loaded at instantiation, not on the fly).
    Args:
        image_size: size of dataset images

    Returns:
        compositions of torchvision transformations
    """
    return transforms.Compose(
        [
            transforms.Resize([int(image_size * 2.0), int(image_size * 2.0)]),
            transforms.ToTensor(),
        ],
    )


def default_mini_imagenet_serving_transform(
    image_size: int, training: bool
) -> Callable:
    """
    Create a composition of torchvision transformations to perform when serving images
     (when data is loaded at instantiation, not on the fly).
    Args:
        image_size: size of dataset images
        training: whether this is a training set or not

    Returns:
        compositions of torchvision transformations
    """
    return (
        transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(**IMAGENET_NORMALIZATION),
            ]
        )
        if training
        else transforms.Compose(
            [
                transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
                transforms.CenterCrop(image_size),
                transforms.Normalize(**IMAGENET_NORMALIZATION),
            ]
        )
    )
