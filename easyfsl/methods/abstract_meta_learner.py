from abc import abstractmethod
from pathlib import Path
from typing import Union

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

        self.best_validation_accuracy = 0.0
        self.best_model_state = None

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

    def fit(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        val_loader: DataLoader = None,
        validation_frequency: int = 1000,
    ):
        """
        Train the model on few-shot classification tasks.
        Args:
            train_loader: loads training data in the shape of few-shot classification tasks
            optimizer: optimizer to train the model
            val_loader: loads data from the validation set in the shape of few-shot classification
                tasks
            validation_frequency: number of training episodes between two validations
        """
        log_update_frequency = 10

        all_loss = []
        self.train()
        with tqdm(
            enumerate(train_loader), total=len(train_loader), desc="Meta-Training"
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

                # Validation
                if val_loader:
                    if episode_index + 1 % validation_frequency == 0:
                        self.validate(val_loader)

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model on the validation set.
        Args:
            val_loader: loads data from the validation set in the shape of few-shot classification
                tasks
        Returns:
            average classification accuracy on the validation set
        """
        validation_accuracy = self.evaluate(val_loader)
        print(f"Validation accuracy: {(100 * validation_accuracy):.2f}%")
        # If this was the best validation performance, we save the model state
        if validation_accuracy > self.best_validation_accuracy:
            print("Best validation accuracy so far!")
            self.best_model_state = self.state_dict()

        return validation_accuracy

    def _check_that_best_state_is_defined(self):
        """
        Will raise an error if self.best_model_state is None, i.e. if no best sate has been
        defined yet.
        """
        if not self.best_model_state:
            raise AttributeError(
                "There is not best state defined for this model. "
                "You need to train the model using validation to define a best state."
            )

    def restore_best_state(self):
        """
        Retrieves the state (i.e. a dictionary of model parameters) of the model at the time it
        obtained its best performance on the validation set.
        """
        self._check_that_best_state_is_defined()
        self.load_state_dict(self.best_model_state)

    def dump_best_state(self, output_path: Union[Path, str]):
        """
        Retrieves the state (i.e. a dictionary of model parameters) of the model at the time it
        obtained its best performance on the validation set.
        Args:
            output_path: path to the output file. Common practice in PyTorch is to save models
                using either a .pt or .pth file extension.
        """
        self._check_that_best_state_is_defined()
        torch.save(self.best_model_state, output_path)
