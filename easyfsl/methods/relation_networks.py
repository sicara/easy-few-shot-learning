"""
See original implementation at
https://github.com/floodsung/LearningToCompare_FSL
"""

import torch
from torch import nn, Tensor
from easyfsl.methods import FewShotClassifier
from easyfsl.utils import compute_prototypes


class RelationNetworks(FewShotClassifier):
    """
    Sung, Flood, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M. Hospedales.
    "Learning to compare: Relation network for few-shot learning." (2018)
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf

    In the Relation Networks algorithm, we first extract feature maps for both support and query
    images. Then we compute the mean of support features for each class (called prototypes).
    To predict the label of a query image, its feature map is concatenated with each class prototype
    and fed into a relation module, i.e. a CNN that outputs a relation score. Finally, the
    classification vector of the query is its relation score to each class prototype.

    Note that for most other few-shot algorithms we talk about feature vectors, because for each
    input image, the backbone outputs a 1-dim feature vector. Here we talk about feature maps,
    because for each input image, the backbone outputs a "feature map" of shape
    (n_channels, width, height). This raises different constraints on the architecture of the
    backbone: while other algorithms require a "flatten" operation in the backbone, here "flatten"
    operations are forbidden.

    Relation Networks use Mean Square Error. This is unusual because this is a classification
    problem. The authors justify this choice by the fact that the output of the model is a relation
    score, which makes it a regression problem. See the article for more details.
    """

    def __init__(self, *args, inner_relation_module_channels: int = 8):
        """
        Build Relation Networks by calling the constructor of AbstractMetaLearner.
        Args:
            *args: all arguments of the init method of AbstractMetaLearner
            inner_relation_module_channels: number of hidden channels between the linear layers of
                the relation module. Defaults to 8.

        Raises:
            ValueError: if the backbone doesn't outputs feature maps, i.e. if its output for a
            given image is not a tensor of shape (n_channels, width, height)
        """
        super().__init__(*args)

        if len(self.backbone_output_shape) != 3:
            raise ValueError(
                "Illegal backbone for Relation Networks. Expected output for an image is a 3-dim "
                "tensor of shape (n_channels, width, height)."
            )

        # Here we build the relation module that will output the relation score for each
        # (query, prototype) pair. See the function docstring for more details.
        self.relation_module = self.build_relation_module(
            inner_relation_module_channels
        )

        # Here we create the field so that the model can store the prototypes for a support set
        self.prototypes = None

    def build_relation_module(self, inner_relation_module_channels: int) -> nn.Module:
        """
        Build the relation module that takes as input the concatenation of two feature
        maps (in our case the feature map of a query and the feature map of a class prototype).
        In order to make the network robust to any change in the dimensions of the input images,
        we made some changes to the architecture defined in the original implementation (typically
        the use of adaptive pooling).
        Args:
            inner_relation_module_channels: number of hidden channels between the linear layers of
                the relaiton module

        Returns:
            the constructed relation module
        """
        return nn.Sequential(
            nn.Sequential(
                nn.Conv2d(
                    self.feature_dimension * 2,
                    self.feature_dimension,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(self.feature_dimension, momentum=1, affine=True),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((5, 5)),
            ),
            nn.Sequential(
                nn.Conv2d(
                    self.feature_dimension,
                    self.feature_dimension,
                    kernel_size=3,
                    padding=0,
                ),
                nn.BatchNorm2d(self.feature_dimension, momentum=1, affine=True),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((1, 1)),
            ),
            nn.Flatten(),
            nn.Linear(self.feature_dimension, inner_relation_module_channels),
            nn.ReLU(),
            nn.Linear(inner_relation_module_channels, 1),
            nn.Sigmoid(),
        )

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Overrides process_support_set of AbstractMetaLearner.
        Extract feature maps from the support set and store class prototypes.

        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """

        support_features = self.backbone(support_images)
        self.prototypes = compute_prototypes(support_features, support_labels)

    def forward(self, query_images: Tensor) -> Tensor:
        """
        Overrides method forward in AbstractMetaLearner.
        Predict the label of a query image by concatenating its feature map with each class
        prototype and feeding the result into a relation module, i.e. a CNN that outputs a relation
        score. Finally, the classification vector of the query is its relation score to each class
        prototype.

        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        query_features = self.backbone(query_images)

        # For each pair (query, prototype), we compute the concatenation of their feature maps
        # Given that query_features is of shape (n_queries, n_channels, width, height), the
        # constructed tensor is of shape (n_queries * n_prototypes, 2 * n_channels, width, height)
        # (2 * n_channels because prototypes and queries are concatenated)
        query_prototype_feature_pairs = torch.cat(
            (
                self.prototypes.unsqueeze(dim=0).expand(
                    query_features.shape[0], -1, -1, -1, -1
                ),
                query_features.unsqueeze(dim=1).expand(
                    -1, self.prototypes.shape[0], -1, -1, -1
                ),
            ),
            dim=2,
        ).view(-1, 2 * self.feature_dimension, *query_features.shape[2:])

        # Each pair (query, prototype) is assigned a relation scores in [0,1]. Then we reshape the
        # tensor so that relation_scores is of shape (n_queries, n_prototypes).
        relation_scores = self.relation_module(query_prototype_feature_pairs).view(
            -1, self.prototypes.shape[0]
        )

        return self.softmax_if_specified(relation_scores)

    @staticmethod
    def is_transductive():
        return False
