import torch
from torch import Tensor, nn

from .finetune import Finetune
from .utils import entropy


class TransductiveFinetuning(Finetune):
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
    have time to solve yet. We welcome contributions.
    As is, it is incompatible with episodic training because we freeze the backbone to perform
    fine-tuning.
    """

    def __init__(
        self,
        *args,
        fine_tuning_steps: int = 25,
        fine_tuning_lr: float = 5e-5,
        temperature: float = 1.0,
        **kwargs,
    ):
        """
        TransductiveFinetuning is very similar to the inductive method Finetune.
        The difference only resides in the way we perform the fine-tuning step and in the
        distance we use. Therefore, we call the super constructor of Finetune
        (and same for preprocess_support_set()).
        Args:
            fine_tuning_steps: number of fine-tuning steps
            fine_tuning_lr: learning rate for fine-tuning
            temperature: temperature applied to the logits before computing
                softmax or cross-entropy. Higher temperature means softer predictions.
        """
        super().__init__(
            *args,
            fine_tuning_steps=fine_tuning_steps,
            fine_tuning_lr=fine_tuning_lr,
            temperature=temperature,
            **kwargs,
        )

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Fine-tune model's parameters based on support classification error and
        query classification entropy.
        """
        query_features = self.compute_features(query_images)

        with torch.enable_grad():
            self.prototypes.requires_grad_()
            optimizer = torch.optim.Adam([self.prototypes], lr=self.fine_tuning_lr)
            for _ in range(self.fine_tuning_steps):
                support_cross_entropy = nn.functional.cross_entropy(
                    self.temperature
                    * self.l2_distance_to_prototypes(self.support_features),
                    self.support_labels,
                )
                query_conditional_entropy = entropy(
                    self.temperature * self.l2_distance_to_prototypes(query_features)
                )
                loss = support_cross_entropy + query_conditional_entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.softmax_if_specified(
            self.l2_distance_to_prototypes(query_features), temperature=self.temperature
        ).detach()

    @staticmethod
    def is_transductive() -> bool:
        return True
