"""
See original implementation (quite far from this one)
at https://github.com/jakesnell/prototypical-networks
"""

import torch
from torch import Tensor

from .few_shot_classifier import FewShotClassifier
from .utils import compute_prototypes


class PrototypicalNetworks(FewShotClassifier):
    """
    Jake Snell, Kevin Swersky, and Richard S. Zemel.
    "Prototypical networks for few-shot learning." (2017)
    https://arxiv.org/abs/1703.05175

    Prototypical networks extract feature vectors for both support and query images. Then it
    computes the mean of support features for each class (called prototypes), and predict
    classification scores for query images based on their euclidean distance to the prototypes.
    """

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract feature vectors from the support set and store class prototypes.

        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """

        support_features = self.backbone.forward(support_images)
        self._raise_error_if_features_are_multi_dimensional(support_features)
        self.prototypes = compute_prototypes(support_features, support_labels)

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Predict query labels based on their distance to class prototypes in the feature space.
        Classification scores are the negative of euclidean distances.

        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        # Extract the features of query images
        query_features = self.backbone.forward(query_images)
        self._raise_error_if_features_are_multi_dimensional(query_features)

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(query_features, self.prototypes)

        # Use it to compute classification scores
        scores = -dists

        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False
