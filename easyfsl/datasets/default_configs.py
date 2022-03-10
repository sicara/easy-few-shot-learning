from torchvision import transforms


IMAGENET_NORMALIZATION = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

DEFAULT_IMAGE_FORMATS = {".bmp", ".png", ".jpeg", ".jpg"}


def default_transforms(image_size: int, training: bool):
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
