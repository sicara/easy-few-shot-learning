import pytest
import torch

from easyfsl.methods import Finetune, TIM, TransductiveFinetuning


class TestFinetuningMethodsRun:
    all_finetuning_methods = [
        Finetune,
        TIM,
        TransductiveFinetuning,
    ]

    @staticmethod
    @pytest.mark.parametrize("method", all_finetuning_methods)
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
        print(model.prototypes)

        model(query_images)

    @staticmethod
    @pytest.mark.parametrize("method", all_finetuning_methods)
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
    @pytest.mark.parametrize("method", all_finetuning_methods)
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
    @pytest.mark.parametrize("method", all_finetuning_methods)
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
