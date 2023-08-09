from torch import Tensor

from .few_shot_classifier import FewShotClassifier
from .utils import compute_prototypes


class SimpleShot(FewShotClassifier):
    """
    Yan Wang, Wei-Lun Chao, Kilian Q. Weinberger, and Laurens van der Maaten.
    "SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning" (2019)
    https://arxiv.org/abs/1911.04623

    Almost exactly Prototypical Classification, but with (optional) centering and cosine distance instead of euclidean distance.
    """
    def __init__(self, 
                 *args, 
                 feature_mean: Tensor = None,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.feature_mean = feature_mean

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract feature vectors from the support set and store class prototypes.
        """

        support_features = self.backbone.forward(support_images)
        self._raise_error_if_features_are_multi_dimensional(support_features)
        if self.feature_mean is not None:
            support_features = support_features - self.feature_mean

        self.prototypes = compute_prototypes(support_features, support_labels)

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
        query_features = self.backbone(query_images)
        self._raise_error_if_features_are_multi_dimensional(query_features)
        if self.feature_mean is not None:
            query_features = query_features - self.feature_mean

        scores = self.cosine_distance_to_prototypes(query_features)

        return self.softmax_if_specified(scores)