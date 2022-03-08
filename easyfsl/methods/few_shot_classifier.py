from abc import abstractmethod

from torch import nn, Tensor

from easyfsl.utils import compute_backbone_output_shape


class FewShotClassifier(nn.Module):
    """
    Abstract class providing methods usable by all few-shot classification algorithms
    """

    def __init__(self, backbone: nn.Module, use_softmax: bool = False):
        """
        Initialize the Few-Shot Classifier
        Args:
            backbone: the feature extractor used by the method. Must output a tensor of the
                appropriate shape (depending on the method)
            use_softmax: whether to return predictions as soft probabilities
        """
        super().__init__()

        self.backbone = backbone
        self.backbone_output_shape = compute_backbone_output_shape(backbone)
        self.feature_dimension = self.backbone_output_shape[0]

        self.use_softmax = use_softmax

    @abstractmethod
    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Predict classification labels.

        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        raise NotImplementedError(
            "All few-shot algorithms must implement a forward method."
        )

    @abstractmethod
    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Harness information from the support set, so that query labels can later be predicted using
        a forward call

        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        raise NotImplementedError(
            "All few-shot algorithms must implement a process_support_set method."
        )

    def softmax_if_specified(self, output: Tensor) -> Tensor:
        """
        If the option is chosen when the classifier is initialized, we perform a softmax on the
        output in order to return soft probabilities.
        Args:
            output: output of the forward method

        Returns:
            output as it was, or output as soft probabilities
        """
        return output.softmax(-1) if self.use_softmax else output
