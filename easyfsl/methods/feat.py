from pathlib import Path
from typing import Union

import torch
from torch import Tensor, nn

from easyfsl.modules import MultiHeadAttention
from easyfsl.modules.feat_resnet12 import feat_resnet12

from .prototypical_networks import PrototypicalNetworks
from .utils import strip_prefix


class FEAT(PrototypicalNetworks):
    """
    Han-Jia Ye, Hexiang Hu, De-Chuan Zhan, Fei Sha.
    "Few-Shot Learning via Embedding Adaptation With Set-to-Set Functions" (CVPR 2020)
    https://openaccess.thecvf.com/content_CVPR_2020/html/Ye_Few-Shot_Learning_via_Embedding_Adaptation_With_Set-to-Set_Functions_CVPR_2020_paper.html

    This method uses an episodically trained attention module to improve the prototypes.
    Queries are then classified based on their euclidean distance to the prototypes,
    as in Prototypical Networks.
    This in an inductive method.

    The attention module must follow specific constraints described in the docstring of FEAT.__init__().
    We provide a default attention module following the one used in the original implementation.
    FEAT can be initialized in the default configuration from the authors, by calling FEAT.from_resnet12_checkpoint().
    """

    def __init__(self, *args, attention_module: nn.Module, **kwargs):
        """
        FEAT needs an additional attention module.
        Args:
            *args:
            attention_module: the forward method must accept 3 Tensor arguments of shape
                (1, num_classes, feature_dimension) and return a pair of Tensor, with the first
                one of shape (1, num_classes, feature_dimension).
                This follows the original implementation of https://github.com/Sha-Lab/FEAT
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.attention_module = attention_module

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Extract prototypes from support set and rectify them with the attention module.
        Args:
            support_images: support images of shape (n_support, **image_shape)
            support_labels: support labels of shape (n_support,)
        """
        super().process_support_set(support_images, support_labels)
        self.prototypes = self.attention_module(
            self.prototypes.unsqueeze(0),
            self.prototypes.unsqueeze(0),
            self.prototypes.unsqueeze(0),
        )[0][0]

    @classmethod
    def from_resnet12_checkpoint(
        cls,
        checkpoint_path: Union[Path, str],
        device: str = "cpu",
        feature_dimension: int = 640,
        use_backbone: bool = True,
        **kwargs,
    ):
        """
        Load a FEAT model from a checkpoint of a resnet12 model as provided by the authors.
        We initialize the default ResNet12 backbone and attention module and load the weights.
        We solve some compatibility issues in the names of the parameters and ensure there
        missing keys.

        Compatible weights can be found here (availability verified 30/05/2023):
            - miniImageNet: https://drive.google.com/file/d/1ixqw1l9XVxl3lh1m5VXkctw6JssahGbQ/view
            - tieredImageNet: https://drive.google.com/file/d/1M93jdOjAn8IihICPKJg8Mb4B-eYDSZfE/view
        Args:
            checkpoint_path: path to the checkpoint
            device: device to load the model on
            feature_dimension: dimension of the features extracted by the backbone.
                Should be 640 with the default Resnet12 backbone.
            use_backbone: if False, we initialize the backbone to nn.Identity() (useful for
                working on pre-extracted features)
        Returns:
            a FEAT model with weights loaded from the checkpoint
        Raises:
            ValueError: if the checkpoint does not contain all the expected keys
                of the backbone or the attention module
        """
        state_dict = torch.load(str(checkpoint_path), map_location=device)["params"]

        if use_backbone:
            backbone = feat_resnet12().to(device)
            backbone_missing_keys, _ = backbone.load_state_dict(
                strip_prefix(state_dict, "encoder."), strict=False
            )
            if len(backbone_missing_keys) > 0:
                raise ValueError(f"Missing keys for backbone: {backbone_missing_keys}")
        else:
            backbone = nn.Identity()

        attention_module = MultiHeadAttention(
            1,
            feature_dimension,
            feature_dimension,
            feature_dimension,
        ).to(device)
        attention_missing_keys, _ = attention_module.load_state_dict(
            strip_prefix(state_dict, "slf_attn."), strict=False
        )
        if len(attention_missing_keys) > 0:
            raise ValueError(
                f"Missing keys for attention module: {attention_missing_keys}"
            )

        return cls(backbone, attention_module=attention_module, **kwargs).to(device)
