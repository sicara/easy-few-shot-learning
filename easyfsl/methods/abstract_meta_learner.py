from abc import abstractmethod

from loguru import logger
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from easyfsl.utils import sliding_average, is_a_feature_extractor


class AbstractMetaLearner(nn.Module):
    """
    Abstract class providing methods usable by all few-shot classification algorithms
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()

        if not is_a_feature_extractor(backbone):
            raise ValueError(
                "Illegal backbone for a few-shot algorithm."
                "Expected output for an image is a 1-dim tensor."
            )

        self.backbone = backbone
        self.criterion = nn.CrossEntropyLoss()

    # pylint: disable=all
    @abstractmethod
    def forward(
        self,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict classification labels.
        Args:
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """
        raise NotImplementedError(
            "All few-shot algorithms must implement a forward method."
        )

    @abstractmethod
    def process_support_set(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ):
        """
        Harness information from the support set, so that query labels can later be predicted using
        a forward call
        Args:
            support_images: images of the support set
            support_labels: labels of support set images
        """
        raise NotImplementedError(
            "All few-shot algorithms must implement a process_support_set method."
        )

    # pylint: enable=all

    def evaluate_on_one_task(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> [int, int]:
        """
        Returns the number of correct predictions of query labels, and the total number of
        predictions.
        """
        self.process_support_set(support_images.cuda(), support_labels.cuda())
        return (
            torch.max(
                self(query_images.cuda()).detach().data,
                1,
            )[1]
            == query_labels.cuda()
        ).sum().item(), len(query_labels)

    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluate the model on few-shot classification tasks
        Args:
            data_loader: loads data in the shape of few-shot classification tasks
        Returns:
            average classification accuracy
        """
        # We'll count everything and compute the ratio at the end
        total_predictions = 0
        correct_predictions = 0

        # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
        # no_grad() tells torch not to keep in memory the whole computational graph
        self.eval()
        with torch.no_grad():
            with tqdm(
                enumerate(data_loader), total=len(data_loader), desc="Evaluation"
            ) as tqdm_eval:
                for _, (
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    _,
                ) in tqdm_eval:
                    correct, total = self.evaluate_on_one_task(
                        support_images, support_labels, query_images, query_labels
                    )

                    total_predictions += total
                    correct_predictions += correct

                    # Log accuracy in real time
                    tqdm_eval.set_postfix(
                        accuracy=correct_predictions / total_predictions
                    )

        return correct_predictions / total_predictions

    def fit_on_task(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        optimizer: optim.Optimizer,
    ) -> float:
        """
        Predict query set labels and updates model's parameters using classification loss
        Args:
            support_images: images of the support set
            support_labels: labels of support set images (used in the forward pass)
            query_images: query set images
            query_labels: labels of query set images (only used for loss computation)
            optimizer: optimizer to train the model

        Returns:
            the value of the classification loss (for reporting purposes)
        """
        optimizer.zero_grad()
        self.process_support_set(support_images.cuda(), support_labels.cuda())
        classification_scores = self(query_images.cuda())

        loss = self.criterion(classification_scores, query_labels.cuda())
        loss.backward()
        optimizer.step()

        return loss.item()

    def fit(self, data_loader: DataLoader, optimizer: optim.Optimizer):
        """
        Train the model on few-shot classification tasks.
        Args:
            data_loader: loads data in the shape of few-shot classification tasks
            optimizer: optimizer to train the model
        """
        log_update_frequency = 10

        all_loss = []
        self.train()
        with tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Meta-Training"
        ) as tqdm_train:
            for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_train:
                loss_value = self.fit_on_task(
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    optimizer,
                )
                all_loss.append(loss_value)

                # Log training loss in real time
                if episode_index % log_update_frequency == 0:
                    tqdm_train.set_postfix(
                        loss=sliding_average(all_loss, log_update_frequency)
                    )
