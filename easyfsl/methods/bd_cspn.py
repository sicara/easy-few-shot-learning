import torch.nn.functional as F
from torch import Tensor

from easyfsl.methods import FewShotClassifier


class BDCSPN(FewShotClassifier):

    """
    Implementation of BD-CSPN (ECCV 2020) https://arxiv.org/abs/1911.10713
    This is a transductive method.
    """

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        self.store_features_labels_and_prototypes(support_images, support_labels)

    def rectify_prototypes(self, query_features: Tensor) -> None:
        n_classes = self.support_labels.unique().size(0)
        one_hot_support_labels = F.one_hot(self.support_labels, n_classes)

        average_support_query_shift = self.support_features.mean(
            0, keepdim=True
        ) - query_features.mean(0, keepdim=True)
        query_features = query_features + average_support_query_shift

        support_logits = self.get_logits_from_cosine_distances_to_prototypes(
            self.support_features
        ).exp()
        query_logits = self.get_logits_from_cosine_distances_to_prototypes(
            query_features
        ).exp()

        one_hot_query_prediction = F.one_hot(query_logits.argmax(-1), n_classes)

        normalization_vector = (
            (one_hot_support_labels * support_logits).sum(0)
            + (one_hot_query_prediction * query_logits).sum(0)
        ).unsqueeze(
            0
        )  # [1, K]
        support_reweighting = (
            one_hot_support_labels * support_logits
        ) / normalization_vector  # [shot_s, K]
        query_reweighting = (
            one_hot_query_prediction * query_logits
        ) / normalization_vector  # [shot_q, K]

        self.prototypes = (support_reweighting * one_hot_support_labels).t().matmul(
            self.support_features
        ) + (query_reweighting * one_hot_query_prediction).t().matmul(query_features)

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        query_features = self.backbone.forward(query_images)

        self.rectify_prototypes(
            query_features=query_features,
        )
        return self.softmax_if_specified(
            self.get_logits_from_cosine_distances_to_prototypes(query_features)
        )

    @staticmethod
    def is_transductive():
        return True
