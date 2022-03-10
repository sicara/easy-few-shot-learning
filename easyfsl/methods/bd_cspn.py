from torch import Tensor, nn

from easyfsl.methods import FewShotClassifier


class BDCSPN(FewShotClassifier):

    """
    Jinlu Liu, Liang Song, Yongqiang Qin
    "Prototype Rectification for Few-Shot Learning" (ECCV 2020)
    https://arxiv.org/abs/1911.10713

    Rectify prototypes with label propagation and feature shifting.
    Classify queries based on their cosine distance to prototypes.
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
        self.store_support_set_data(support_images, support_labels)

    def rectify_prototypes(self, query_features: Tensor):
        """
        Updates prototypes with label propagation and feature shifting.
        Args:
            query_features: query features
        """
        n_classes = self.support_labels.unique().size(0)
        one_hot_support_labels = nn.functional.one_hot(self.support_labels, n_classes)

        average_support_query_shift = self.support_features.mean(
            0, keepdim=True
        ) - query_features.mean(0, keepdim=True)
        query_features = query_features + average_support_query_shift

        support_logits = self.cosine_distance_to_prototypes(self.support_features).exp()
        query_logits = self.cosine_distance_to_prototypes(query_features).exp()

        one_hot_query_prediction = nn.functional.one_hot(
            query_logits.argmax(-1), n_classes
        )

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
        """
        Overrides forward method of FewShotClassifier.
        Updates prototypes using query images, then classify query images based
        on their cosine distance to updated prototypes.
        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        query_features = self.backbone.forward(query_images)

        self.rectify_prototypes(
            query_features=query_features,
        )
        return self.softmax_if_specified(
            self.cosine_distance_to_prototypes(query_features)
        )

    @staticmethod
    def is_transductive() -> bool:
        return True
