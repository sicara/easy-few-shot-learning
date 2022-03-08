from typing import Tuple

import torch.nn.functional as F
from torch import Tensor

from easyfsl.methods import FewShotClassifier
from easyfsl.utils import compute_prototypes


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
        Kes = self.support_labels.unique().size(0)
        one_hot_s = F.one_hot(self.support_labels, Kes)  # [shot_s, K]
        eta = self.support_features.mean(0, keepdim=True) - query_features.mean(
            0, keepdim=True
        )  # [1, feature_dim]
        query_features = query_features + eta

        logits_s = self.get_logits_from_cosine_distances_to_prototypes(
            self.support_features
        ).exp()  # [shot_s, K]
        logits_q = self.get_logits_from_cosine_distances_to_prototypes(
            query_features
        ).exp()  # [shot_q, K]

        preds_q = logits_q.argmax(-1)
        one_hot_q = F.one_hot(preds_q, Kes)

        normalization = (
            (one_hot_s * logits_s).sum(0) + (one_hot_q * logits_q).sum(0)
        ).unsqueeze(
            0
        )  # [1, K]
        w_s = (one_hot_s * logits_s) / normalization  # [shot_s, K]
        w_q = (one_hot_q * logits_q) / normalization  # [shot_q, K]

        self.prototypes = (w_s * one_hot_s).t().matmul(self.support_features) + (
            w_q * one_hot_q
        ).t().matmul(query_features)

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
