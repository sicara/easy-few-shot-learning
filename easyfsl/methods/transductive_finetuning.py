import torch
from torch import Tensor, nn


from easyfsl.methods import FewShotClassifier
from easyfsl.utils import entropy


class TransductiveFinetuning(FewShotClassifier):
    """
    Guneet S. Dhillon, Pratik Chaudhari, Avinash Ravichandran, Stefano Soatto.
    "A Baseline for Few-Shot Image Classification" (ICLR 2020)
    https://arxiv.org/abs/1909.02729

    Fine-tune the parameters of the pre-trained model based on
        1) classification error on support images
        2) classification entropy for query images
    Classify queries based on their euclidean distance to prototypes.
    This is a transductive method.
    WARNING: this implementation only updates prototypes, not the whole set of model's
    parameters. Updating the model's parameters raises performance issues that we didn't
    As is, it is incompatible with episodic training because we freeze the backbone to perform
    fine-tuning.

    have time to solve yet.
    """

    def __init__(
        self,
        *args,
        fine_tuning_steps: int = 25,
        fine_tuning_lr: float = 5e-5,
        **kwargs,
    ):
        """
        Args:
            fine_tuning_steps: number of fine-tuning steps
            fine_tuning_lr: learning rate for fine-tuning
        """
        super().__init__(*args, **kwargs)

        # Since we fine-tune the prototypes we need to make them leaf variables
        # i.e. we need to freeze the backbone.
        self.backbone.requires_grad_(False)

        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr

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
        self.store_support_set_data(support_images, support_labels)

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Fine-tune model's parameters based on support classification error and
        query classification entropy.
        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        query_features = self.backbone.forward(query_images)

        with torch.enable_grad():
            self.prototypes.requires_grad_()
            optimizer = torch.optim.Adam([self.prototypes], lr=self.fine_tuning_lr)
            for _ in range(self.fine_tuning_steps):

                support_cross_entropy = nn.functional.cross_entropy(
                    self.l2_distance_to_prototypes(self.support_features),
                    self.support_labels,
                )
                query_conditional_entropy = entropy(
                    self.l2_distance_to_prototypes(query_features)
                )
                loss = support_cross_entropy + query_conditional_entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.softmax_if_specified(
            self.l2_distance_to_prototypes(query_features)
        ).detach()

    @staticmethod
    def is_transductive() -> bool:
        return True
