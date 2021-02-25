"""
See original implementation (quite far from this one)
at https://github.com/jakesnell/prototypical-networks
"""

import torch

from easyfsl.methods import AbstractMetaLearner


class PrototypicalNetworks(AbstractMetaLearner):
    """
    Snell, Jake, Kevin Swersky, and Richard S. Zemel. "Prototypical networks for few-shot learning."
    https://arxiv.org/abs/1703.05175

    Prototypical networks extract feature vectors for both support and query images. Then it
    computes the mean of support features for each class (called prototypes), and predict
    classification scores for query images based on their distance to the prototypes.
    """

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Overwrites forward method of AbstractFewShotAlgorithm.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # Use it to compute classification scores
        scores = -dists
        return scores
