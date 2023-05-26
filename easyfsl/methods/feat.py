from torch import Tensor, nn

from .prototypical_networks import PrototypicalNetworks


# TODO: fix how-to in docstring
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

    ```python
        from easyfsl.modules import MultiHeadAttention

        attention_module = MultiHeadAttention(
            1,
            feature_dimension,
            feature_dimension,
            feature_dimension,
        )
        attention_module.load_state_dict(path_to_downloaded_state_dict)

        model = FEAT(backbone, attention_module=attention_module)
    ```
    """

    def __init__(self, *args, attention_module: nn.Module, **kwargs):
        """
        FEAT needs an additional attention module.
        Args:
            *args:
            attention_module: the forward method must accept 3 Tensor arguments of shape
                (1, num_classes, feature_dimension) and return a pair of Tensor, with the first
                one of shape (1, num_classes, feature_dimension).
                This follows the original implementation of
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.attention_module = attention_module

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        super().process_support_set(support_images, support_labels)
        self.prototypes = self.attention_module(
            self.prototypes.unsqueeze(0),
            self.prototypes.unsqueeze(0),
            self.prototypes.unsqueeze(0),
        )[0][0]
