from pathlib import Path
from typing import Union, Callable

import torch
from torch import Tensor
from torchvision.datasets import ImageFolder

from easyfsl.datasets.default_configs import default_transform


NOT_A_TENSOR_ERROR_MESSAGE = (
    "SupportSetFolder handles instances as tensors. "
    "Please ensure that the specific transform outputs a tensor."
)


class SupportSetFolder(ImageFolder):
    """
    Create a support set from images located in a specified folder
    with the following file structure:

    root:
      |_ subfolder_1:
             |_ image_1
             |_  …
             |_ image_n
      |_ subfolder_2:
             |_ image_1
             |_  …
             |_ image_n

    Following the ImageFolder logic, images of a same subfolder will share the same label,
    and the classes will be named after the subfolders.

    Example of use:

    predict_transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    support_set = SupportSetFolder(
        root=path_to_support_images,
        transform=predict_transformation,
        device="cuda"
    )
    with torch.no_grad():
        few_shot_classifier.eval()
        few_shot_classifier.process_support_set(support_set.get_images(), support_set.get_labels())
        class_names = support_set.classes
        predicted_labels = few_shot_classifier(query_images.to(device)).argmax(dim=1)
        predicted_classes = [ support_set.classes[label] for label in predicted_labels]
    """

    def __init__(
        self,
        root: Union[str, Path],
        device="cpu",
        image_size: int = 84,
        transform: Callable = None,
        **kwargs
    ):
        """
        Args:
            device:
            **kwargs: kwargs for the parent ImageFolder class
        """
        transform = (
            transform if transform else default_transform(image_size, training=False)
        )

        super().__init__(str(root), transform=transform, **kwargs)

        self.device = device
        try:
            self.images = torch.stack([instance[0] for instance in self]).to(
                self.device
            )
        except TypeError as type_error:
            raise TypeError(NOT_A_TENSOR_ERROR_MESSAGE) from type_error

    def get_images(self) -> Tensor:
        """
        Returns:
            support set images as a (n_images, n_channels, width, height) tensor
                on the selected device
        """
        return self.images

    def get_labels(self) -> Tensor:
        """
        Returns:
            support set labels as a tensor on the selected device
        """
        return torch.tensor(self.targets).to(self.device)
