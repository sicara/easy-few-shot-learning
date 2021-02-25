from abc import abstractmethod

from loguru import logger
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from easyfsl.utils import sliding_average


class AbstractMetaLearner(nn.Module):
    """
    Abstract class providing methods usable by all few-shot classification algorithms
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.criterion = nn.CrossEntropyLoss()

    # pylint: disable=all
    @abstractmethod
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            support_images: images of the support set
            support_labels: labels of support set images
            query_images: images of the query set
        Returns:
            a prediction of classification scores for query images
        """

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
        return (
            torch.max(
                self(support_images.cuda(), support_labels.cuda(), query_images.cuda())
                .detach()
                .data,
                1,
            )[1]
            == query_labels.cuda()
        ).sum().item(), len(query_labels)

    def evaluate(self, data_loader: DataLoader):
        """
        Evaluate the model on few-shot classification tasks
        Args:
            data_loader: loads data in the shape of few-shot classification tasks
        """
        # We'll count everything and compute the ratio at the end
        total_predictions = 0
        correct_predictions = 0

        # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
        # no_grad() tells torch not to keep in memory the whole computational graph
        self.eval()
        logger.info("Starting model evaluation...")
        with torch.no_grad():
            for _, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm(enumerate(data_loader), total=len(data_loader)):
                correct, total = self.evaluate_on_one_task(
                    support_images, support_labels, query_images, query_labels
                )

                total_predictions += total
                correct_predictions += correct

        logger.info(
            f"Model tested on {len(data_loader)} tasks. "
            "Accuracy: {(100 * correct_predictions / total_predictions):.2f}%"
        )

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
        classification_scores = self(
            support_images.cuda(), support_labels.cuda(), query_images.cuda()
        )

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
        logger.info("Starting meta-training ...")
        with tqdm(enumerate(data_loader), total=len(data_loader)) as tqdm_train:
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

                if episode_index % log_update_frequency == 0:
                    tqdm_train.set_postfix(
                        loss=sliding_average(all_loss, log_update_frequency)
                    )
