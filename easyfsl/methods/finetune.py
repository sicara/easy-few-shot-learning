import torch
from torch import Tensor, nn

from .few_shot_classifier import FewShotClassifier


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
        fine_tuning_steps: int = 200,
        fine_tuning_lr: float = 1e-4,
        temperature: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            fine_tuning_steps: number of fine-tuning steps
            fine_tuning_lr: learning rate for fine-tuning
            temperature: temperature applied to the logits before computing
                softmax or cross-entropy. Higher temperature means softer predictions.
        """
        super().__init__(*args, **kwargs)

        # Since we fine-tune the prototypes we need to make them leaf variables
        # i.e. we need to freeze the backbone.
        self.backbone.requires_grad_(False)

        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr
        self.temperature = temperature

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Fine-tune prototypes based on support classification error.
        Then classify w.r.t. to cosine distance to prototypes.
        """
        query_features = self.compute_features(query_images)

        with torch.enable_grad():
            self.prototypes.requires_grad_()
            optimizer = torch.optim.Adam([self.prototypes], lr=self.fine_tuning_lr)
            for _ in range(self.fine_tuning_steps):
                support_logits = self.cosine_distance_to_prototypes(
                    self.support_features
                )
                loss = nn.functional.cross_entropy(
                    self.temperature * support_logits, self.support_labels
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.softmax_if_specified(
            self.cosine_distance_to_prototypes(query_features),
            temperature=self.temperature,
        ).detach()

    @staticmethod
    def is_transductive() -> bool:
        return False
