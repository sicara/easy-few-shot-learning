from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor

from easyfsl.methods import FewShotClassifier
from easyfsl.utils import compute_prototypes


class TIM(FewShotClassifier):
    """
    Implementation of the Transductive Information Maximization method (NeurIPS 2020)
    https://arxiv.org/abs/2008.11297
    TIM is a transductive method.
    """

    def __init__(
        self,
        fine_tuning_steps: int = 100,
        fine_tuning_lr: float = 1e-3,
        loss_weights: List[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_weights = [1.0, 1.0, 0.1] if loss_weights is None else loss_weights
        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr

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

        # Metric dic
        num_classes = self.support_labels.unique().size(0)
        support_labels_one_hot = F.one_hot(self.support_labels, num_classes)

        # Run adaptation
        self.prototypes.requires_grad_()
        optimizer = torch.optim.Adam([self.prototypes], lr=self.fine_tuning_lr)

        for i in range(self.fine_tuning_steps):
            logits_s = self.get_logits_from_euclidean_distances_to_prototypes(
                self.support_features
            )
            logits_q = self.get_logits_from_euclidean_distances_to_prototypes(
                query_features
            )

            ce = -(support_labels_one_hot * logits_s.log_softmax(1)).sum(1).mean(0)
            q_probs = logits_q.softmax(1)
            q_cond_ent = -(q_probs * torch.log(q_probs + 1e-12)).sum(1).mean(0)
            marginal_y = q_probs.mean(0)
            q_ent = -(marginal_y * torch.log(marginal_y)).sum(0)

            loss = self.loss_weights[0] * ce - (
                self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self.softmax_if_specified(logits_q).detach()

    @staticmethod
    def is_transductive():
        return False
