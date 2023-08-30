import torch
from torch import Tensor, nn

from easyfsl.methods.utils import power_transform

from .few_shot_classifier import FewShotClassifier

MAXIMUM_SINKHORN_ITERATIONS = 1000


class PTMAP(FewShotClassifier):
    """
    Yuqing Hu, Vincent Gripon, Stéphane Pateux.
    "Leveraging the Feature Distribution in Transfer-based Few-Shot Learning" (2020)
    https://arxiv.org/abs/2006.03806

    Query soft assignments are computed as the optimal transport plan to class prototypes.
    At each iteration, prototypes are fine-tuned based on the soft assignments.
    This is a transductive method.
    """

    def __init__(
        self,
        *args,
        fine_tuning_steps: int = 10,
        fine_tuning_lr: float = 0.2,
        lambda_regularization: float = 10.0,
        power_factor: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr
        self.lambda_regularization = lambda_regularization
        self.power_factor = power_factor

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Predict query soft assignments following Algorithm 1 of the paper.
        """
        query_features = self.compute_features(query_images)

        support_assignments = nn.functional.one_hot(
            self.support_labels, len(self.prototypes)
        )
        for _ in range(self.fine_tuning_steps):
            query_soft_assignments = self.compute_soft_assignments(query_features)
            all_features = torch.cat([self.support_features, query_features], 0)
            all_assignments = torch.cat(
                [support_assignments, query_soft_assignments], dim=0
            )

            self.update_prototypes(all_features, all_assignments)

        return self.compute_soft_assignments(query_features)

    def compute_features(self, images: Tensor) -> Tensor:
        """
        Apply power transform on features following Equation (1) in the paper.
        Args:
            images: images of shape (n_images, **image_shape)
        Returns:
            features of shape (n_images, feature_dimension) with power-transform.
        """
        features = super().compute_features(images)
        return power_transform(features, self.power_factor)

    def compute_soft_assignments(self, query_features: Tensor) -> Tensor:
        """
        Compute soft assignments from queries to prototypes, following Equation (3) of the paper.
        Args:
            query_features: query features, of shape (n_queries, feature_dim)

        Returns:
            soft assignments from queries to prototypes, of shape (n_queries, n_classes)
        """

        distances_to_prototypes = (
            torch.cdist(query_features, self.prototypes) ** 2
        )  # [Nq, K]

        soft_assignments = self.compute_optimal_transport(
            distances_to_prototypes, epsilon=1e-6
        )

        return soft_assignments

    def compute_optimal_transport(
        self, cost_matrix: Tensor, epsilon: float = 1e-6
    ) -> Tensor:
        """
        Compute the optimal transport plan from queries to prototypes using Sinkhorn-Knopp algorithm.
        Args:
            cost_matrix: euclidean distances from queries to prototypes,
                of shape (n_queries, n_classes)
            epsilon: convergence parameter. Stop when the update is smaller than epsilon.
        Returns:
            transport plan from queries to prototypes of shape (n_queries, n_classes)
        """

        instance_multiplication_factor = cost_matrix.shape[0] // cost_matrix.shape[1]

        transport_plan = torch.exp(-self.lambda_regularization * cost_matrix)
        transport_plan /= transport_plan.sum(dim=(0, 1), keepdim=True)

        for _ in range(MAXIMUM_SINKHORN_ITERATIONS):
            per_class_sums = transport_plan.sum(1)
            transport_plan *= (1 / (per_class_sums + 1e-10)).unsqueeze(1)
            transport_plan *= (
                instance_multiplication_factor / (transport_plan.sum(0) + 1e-10)
            ).unsqueeze(0)
            if torch.max(torch.abs(per_class_sums - transport_plan.sum(1))) < epsilon:
                break

        return transport_plan

    def update_prototypes(self, all_features, all_assignments) -> None:
        """
        Update prototypes by weigh-averaging the features with their soft assignments,
            following Equation (6) of the paper.
        Args:
            all_features: concatenation of support and query features,
                of shape (n_support + n_query, feature_dim)
            all_assignments: concatenation of support and query soft assignments,
                of shape (n_support + n_query, n_classes)-
        """
        new_prototypes = (all_assignments.T @ all_features) / all_assignments.sum(
            0
        ).unsqueeze(1)
        delta = new_prototypes - self.prototypes
        self.prototypes += self.fine_tuning_lr * delta

    @staticmethod
    def is_transductive() -> bool:
        return True
