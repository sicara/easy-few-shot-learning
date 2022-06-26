from itertools import product

import pytest
import torch

from easyfsl.datasets import SupportSetFolder
from easyfsl.methods import Finetune, TIM, TransductiveFinetuning

ALL_FINETUNING_METHODS = [
    Finetune,
    TIM,
    TransductiveFinetuning,
]


class TestFinetuningMethodsRun:
    @staticmethod
    @pytest.mark.parametrize("method", ALL_FINETUNING_METHODS)
    def test_methods_run_in_ordinary_context(
        method, example_few_shot_classification_task, dummy_network
    ):
        model = method(backbone=dummy_network, fine_tuning_steps=2)
        (
            support_images,
            support_labels,
            query_images,
        ) = example_few_shot_classification_task

        model.process_support_set(support_images, support_labels)

        model(query_images)

    @staticmethod
    @pytest.mark.parametrize("method", ALL_FINETUNING_METHODS)
    def test_methods_run_in_no_grad_context(
        method, example_few_shot_classification_task, dummy_network
    ):
        model = method(backbone=dummy_network, fine_tuning_steps=2)
        (
            support_images,
            support_labels,
            query_images,
        ) = example_few_shot_classification_task
        with torch.no_grad():
            model.process_support_set(support_images, support_labels)

            model(query_images)

    @staticmethod
    @pytest.mark.parametrize("method", ALL_FINETUNING_METHODS)
    def test_prototypes_update_in_ordinary_context(
        method, example_few_shot_classification_task, dummy_network
    ):
        model = method(backbone=dummy_network, fine_tuning_steps=2, fine_tuning_lr=1.0)
        (
            support_images,
            support_labels,
            query_images,
        ) = example_few_shot_classification_task
        model.process_support_set(support_images, support_labels)
        prototypes = model.prototypes.clone()

        model(query_images)
        assert not prototypes.isclose(model.prototypes, atol=1e-02).all()

    @staticmethod
    @pytest.mark.parametrize("method", ALL_FINETUNING_METHODS)
    def test_prototypes_update_in_no_grad_context(
        method, example_few_shot_classification_task, dummy_network
    ):
        model = method(backbone=dummy_network, fine_tuning_steps=2, fine_tuning_lr=1.0)
        (
            support_images,
            support_labels,
            query_images,
        ) = example_few_shot_classification_task
        with torch.no_grad():
            model.process_support_set(support_images, support_labels)
            prototypes = model.prototypes.clone()

            model(query_images)
            assert not prototypes.isclose(model.prototypes, atol=1e-02).all()


class TestFinetuningMethodsCanProcessSupportSetFolder:
    @staticmethod
    @pytest.mark.parametrize("method", ALL_FINETUNING_METHODS)
    def test_finetuning_methods_can_process_support_set_from_balanced_folder(
        method, dummy_network
    ):
        support_set = SupportSetFolder(
            "easyfsl/tests/datasets/resources/balanced_support_set"
        )
        support_images = support_set.get_images()
        support_labels = support_set.get_labels()

        model = method(backbone=dummy_network, fine_tuning_steps=2, fine_tuning_lr=1.0)
        model.process_support_set(support_images, support_labels)

        query_images = torch.randn((4, 3, 224, 224))
        model(query_images)

    @staticmethod
    @pytest.mark.parametrize("method", ALL_FINETUNING_METHODS)
    def test_finetuning_methods_can_process_support_set_from_unbalanced_folder(
        method, dummy_network
    ):
        support_set = SupportSetFolder(
            "easyfsl/tests/datasets/resources/unbalanced_support_set"
        )
        support_images = support_set.get_images()
        support_labels = support_set.get_labels()

        model = method(backbone=dummy_network, fine_tuning_steps=2, fine_tuning_lr=1.0)
        model.process_support_set(support_images, support_labels)

        query_images = torch.randn((4, 3, 224, 224))
        model(query_images)

    @staticmethod
    @pytest.mark.parametrize(
        ("method", "support_set_path"),
        list(
            product(
                ALL_FINETUNING_METHODS,
                [
                    "easyfsl/tests/datasets/resources/unbalanced_support_set",
                    "easyfsl/tests/datasets/resources/balanced_support_set",
                ],
            )
        ),
    )
    def test_finetuning_methods_store_correct_support_labels(
        method, support_set_path, dummy_network
    ):
        support_set = SupportSetFolder(support_set_path)
        support_images = support_set.get_images()
        support_labels = support_set.get_labels()

        model = method(backbone=dummy_network, fine_tuning_steps=2, fine_tuning_lr=1.0)
        model.process_support_set(support_images, support_labels)

        assert torch.equal(model.support_labels, support_labels)

    @staticmethod
    @pytest.mark.parametrize(
        (
            "method",
            "support_set_path_and_expected_prototypes",
        ),
        list(
            product(
                ALL_FINETUNING_METHODS,
                [
                    (
                        "easyfsl/tests/datasets/resources/unbalanced_support_set",
                        [-0.0987, -0.0489, -0.3414],
                    ),
                    (
                        "easyfsl/tests/datasets/resources/balanced_support_set",
                        [-0.0987, 0.2805, -0.3582],
                    ),
                ],
            )
        ),
    )
    def test_finetuning_methods_store_correct_prototypes(
        method, support_set_path_and_expected_prototypes, deterministic_dummy_network
    ):
        support_set_path, expected_prototypes = support_set_path_and_expected_prototypes
        support_set = SupportSetFolder(support_set_path)
        support_images = support_set.get_images()
        support_labels = support_set.get_labels()

        model = method(
            backbone=deterministic_dummy_network,
            fine_tuning_steps=2,
            fine_tuning_lr=1.0,
        )
        model.process_support_set(support_images, support_labels)

        assert torch.all(
            torch.isclose(
                model.prototypes,
                torch.tensor(expected_prototypes).unsqueeze(1),
                atol=1e-04,
            )
        )
