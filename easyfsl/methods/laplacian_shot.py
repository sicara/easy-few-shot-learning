import torch
from torch import Tensor

from .bd_cspn import BDCSPN
from .utils import k_nearest_neighbours


class LaplacianShot(BDCSPN):
    """
    Imtiaz Masud Ziko, Jose Dolz, Eric Granger, Ismail Ben Ayed.
    "Laplacian Regularized Few-Shot Learning" (ICML 2020)
    https://arxiv.org/abs/2006.15486

    LaplacianShot updates the soft-assignments using a Laplacian Regularization to
    improve consistency between the assignments of neighbouring query points.
    Default hyperparameters have been optimized for 5-way 5-shot classification on
    miniImageNet (see https://github.com/ebennequin/few-shot-open-set/blob/master/configs/classifiers.yaml).

    LaplianShot is a transductive method.
    """

    def __init__(
        self,
        *args,
        inference_steps: int = 20,
        knn: int = 3,
        lambda_regularization: float = 0.7,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.knn = knn
        self.inference_steps = inference_steps
        self.lambda_regularization = lambda_regularization

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        query_features = self.compute_features(query_images)
        self.rectify_prototypes(query_features=query_features)

        features_to_prototypes_distances = (
            torch.cdist(query_features, self.prototypes) ** 2
        )
        pairwise_affinities = self.compute_pairwise_affinities(query_features)
        predictions = self.bound_updates(
            initial_scores=features_to_prototypes_distances, kernel=pairwise_affinities
        )

        return predictions

    def compute_pairwise_affinities(self, features: Tensor) -> Tensor:
        """
        Build pairwise affinity matrix from features using k-nearest neighbours.
        Item (i, j) of the matrix is 1 if i is among the k-nearest neighbours of j, and vice versa, and 0 otherwise.
        Args:
            features: tensor of shape (n_features, feature_dimension)

        Returns:
            tensor of shape (n_features, n_features) corresponding to W in the paper.
        """
        # Compute the k-nearest neighbours of each feature vector.
        # Each row is the indices of the k nearest neighbours of the corresponding feature, not including itself
        nearest_neighbours = k_nearest_neighbours(features, self.knn)
        affinity_matrix = torch.zeros((len(features), len(features))).to(
            nearest_neighbours.device
        )
        for vector_index, vector_nearest_neighbours in enumerate(nearest_neighbours):
            affinity_matrix[vector_index].index_fill_(0, vector_nearest_neighbours, 1)

        return affinity_matrix

    def compute_upper_bound(
        self, soft_assignments: Tensor, initial_scores: Tensor, kernel: Tensor
    ) -> float:
        """
        Compute the upper bound objective for the soft assignments following Equation (7) of the paper.
        Args:
            soft_assignments: soft assignments of shape (n_query, n_classes), $$y_q$$ in the paper
            initial_scores: distances from each query to each prototype,
                of shape (n_query, n_classes), $$a_q$$ in the paper
            kernel: pairwise affinities between query feature vectors,
                of shape (n_features, n_features), $$W$$ in the paper
        Returns:
            upper bound objective
        """
        pairwise = kernel.matmul(soft_assignments)
        temp = (initial_scores * soft_assignments) + (
            -self.lambda_regularization * pairwise * soft_assignments
        )
        upper_bound = (soft_assignments * (soft_assignments + 1e-12).log() + temp).sum()

        return upper_bound.item()

    def bound_updates(self, initial_scores: Tensor, kernel: Tensor) -> Tensor:
        """
        Compute the soft assignments using the bound update algorithm described in the paper
        as Algorithm 1.
        Args:
            initial_scores: distances from each query to each prototype, of shape (n_query, n_classes)
            kernel: pairwise affinities between query feature vectors, of shape (n_features, n_features)
        Returns:
            soft_assignments: soft assignments of shape (n_query, n_classes)
        """
        old_upper_bound = float("inf")
        soft_assignments = (-initial_scores).softmax(dim=1)
        for i in range(self.inference_steps):
            additive = -initial_scores
            mul_kernel = kernel.matmul(soft_assignments)
            soft_assignments = -self.lambda_regularization * mul_kernel
            additive = additive - soft_assignments
            soft_assignments = additive.softmax(dim=1)
            upper_bound = self.compute_upper_bound(
                soft_assignments, initial_scores, kernel
            )

            if i > 1 and (
                abs(upper_bound - old_upper_bound) <= 1e-6 * abs(old_upper_bound)
            ):
                break

            old_upper_bound = upper_bound

        return soft_assignments

    @staticmethod
    def is_transductive() -> bool:
        return True
