"""
See original implementation (quite far from this one)
at https://github.com/jakesnell/prototypical-networks
"""

import torch

from easyfsl.methods import AbstractMetaLearner
from easyfsl.utils import compute_prototypes


class PrototypicalNetworks(AbstractMetaLearner):
    """
    Snell, Jake, Kevin Swersky, and Richard S. Zemel. "Prototypical networks for few-shot learning."
    https://arxiv.org/abs/1703.05175

    Prototypical networks extract feature vectors for both support and query images. Then it
    computes the mean of support features for each class (called prototypes), and predict
    classification scores for query images based on their distance to the prototypes.
    """

    def __init__(self, *args):
        super().__init__(*args)

        self.prototypes = None

    def process_support_set(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ):
        """
        Overwrites process_support_set of AbstractMetaLearner. Extract features from the support set
        and store class prototypes
        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """

        support_features = self.backbone.forward(support_images)
        self.prototypes = compute_prototypes(support_features, support_labels)

    def forward(
        self,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Overwrites forward method of AbstractMetaLearner. Predict query labels based on their
        distance to class prototypes in the feature space.
        """
        # Extract the features of support and query images
        z_query = self.backbone.forward(query_images)

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, self.prototypes)

        # Use it to compute classification scores
        scores = -dists
        return scores
