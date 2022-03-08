from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


from easyfsl.methods import FewShotClassifier
from easyfsl.utils import compute_prototypes


class Finetune(FewShotClassifier):
    """
    Implementation of Finetune (or Baseline method) (ICLR 2019) https://arxiv.org/abs/1904.04232
    This is an inductive method.
    """

    def __init__(
        self,
        fine_tuning_steps: int = 10,
        fine_tuning_lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fine_tuning_steps = fine_tuning_steps
        self.lr = fine_tuning_lr

        self.prototypes = None
        self.support_features = None
        self.support_labels = None

    def process_support_set(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        self.store_features_labels_and_prototypes(support_images, support_labels)

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:

        query_features = self.backbone.forward(query_images)

        # Run adaptation
        self.prototypes.requires_grad_()
        optimizer = torch.optim.Adam([self.prototypes], lr=self.lr)
        for i in range(self.fine_tuning_steps):

            logits_s = self.get_logits_from_cosine_distances_to_prototypes(
                self.support_features
            )
            ce = nn.functional.cross_entropy(logits_s, self.support_labels)
            optimizer.zero_grad()
            ce.backward()
            optimizer.step()

        return self.softmax_if_specified(
            self.get_logits_from_cosine_distances_to_prototypes(query_features)
        ).detach()

    @staticmethod
    def is_transductive():
        return False
