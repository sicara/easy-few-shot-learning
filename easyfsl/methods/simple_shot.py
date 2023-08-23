from torch import Tensor

from .few_shot_classifier import FewShotClassifier


class SimpleShot(FewShotClassifier):
    """
    Yan Wang, Wei-Lun Chao, Kilian Q. Weinberger, and Laurens van der Maaten.
    "SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning" (2019)
    https://arxiv.org/abs/1911.04623

    Almost exactly Prototypical Classification, but with cosine distance instead of euclidean distance.
    """

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Predict classification labels.
        Args:
            query_images: images of the query set of shape (n_query, **image_shape)
        Returns:
            a prediction of classification scores for query images of shape (n_query, n_classes)
        """
        query_features = self.compute_features(query_images)
        self._raise_error_if_features_are_multi_dimensional(query_features)

        scores = self.cosine_distance_to_prototypes(query_features)

        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False
