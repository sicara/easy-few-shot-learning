import torch
from torch import Tensor, nn


from easyfsl.methods import FewShotClassifier


class Finetune(FewShotClassifier):
    """
    Wei-Yu Chen, Yen-Cheng Liu, Zsolt Kira, Yu-Chiang Frank Wang, Jia-Bin Huang
    A Closer Look at Few-shot Classification (ICLR 2019)
    https://arxiv.org/abs/1904.04232

    Fine-tune prototypes based on classification error on support images.
    Classify queries based on their cosine distances to updated prototypes.
    As is, it is incompatible with episodic training because we freeze the backbone to perform
    fine-tuning.

    This is an inductive method.
    """

    def __init__(
        self,
        *args,
        fine_tuning_steps: int = 10,
        fine_tuning_lr: float = 1e-3,
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
        Fine-tune prototypes based on support classification error.
        Then classify w.r.t. to cosine distance to prototypes.
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

                support_logits = self.cosine_distance_to_prototypes(
                    self.support_features
                )
                loss = nn.functional.cross_entropy(support_logits, self.support_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.softmax_if_specified(
            self.cosine_distance_to_prototypes(query_features)
        ).detach()

    @staticmethod
    def is_transductive() -> bool:
        return False
