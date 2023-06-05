import pytest
import torch
from torch import nn

from easyfsl.methods import PrototypicalNetworks
from easyfsl.methods.feat import FEAT
from easyfsl.modules.attention_modules import MultiHeadAttention

FLATTENED_IMAGE_SIZE = 3072


class TestFeat:
    @staticmethod
    def test_forward_runs_without_error(example_few_shot_classification_task):
        (
            support_images,
            support_labels,
            query_images,
        ) = example_few_shot_classification_task

        model = FEAT(
            nn.Flatten(),
            attention_module=MultiHeadAttention(
                1, FLATTENED_IMAGE_SIZE, FLATTENED_IMAGE_SIZE, FLATTENED_IMAGE_SIZE
            ),
        )
        model.eval()
        model.process_support_set(support_images, support_labels)
        model(query_images)

    @staticmethod
    def test_returns_expected_output_for_example_images(
        example_few_shot_classification_task,
    ):
        (
            support_images,
            support_labels,
            query_images,
        ) = example_few_shot_classification_task

        torch.manual_seed(1)
        torch.set_num_threads(1)

        model = FEAT(
            nn.Flatten(),
            attention_module=MultiHeadAttention(
                1, FLATTENED_IMAGE_SIZE, FLATTENED_IMAGE_SIZE, FLATTENED_IMAGE_SIZE
            ),
        )
        model.eval()

        model.process_support_set(support_images, support_labels)
        predictions = model(query_images)

        assert torch.all(
            torch.isclose(
                predictions,
                torch.tensor(
                    [[-59.5840, -57.9814], [-71.3547, -70.1513]],
                ),
                atol=1,
            )
        )

    @staticmethod
    def test_raise_error_when_features_are_not_1_dim(
        example_few_shot_classification_task,
    ):
        (
            support_images,
            support_labels,
            _,
        ) = example_few_shot_classification_task

        model = FEAT(
            nn.Identity(),
            attention_module=MultiHeadAttention(
                1, FLATTENED_IMAGE_SIZE, FLATTENED_IMAGE_SIZE, FLATTENED_IMAGE_SIZE
            ),
        )
        with pytest.raises(ValueError):
            model.process_support_set(support_images, support_labels)

    @staticmethod
    def test_attention_module_updates_prototypes(example_few_shot_classification_task):
        (
            support_images,
            support_labels,
            _,
        ) = example_few_shot_classification_task
        model_feat = FEAT(
            nn.Flatten(),
            attention_module=MultiHeadAttention(
                1, FLATTENED_IMAGE_SIZE, FLATTENED_IMAGE_SIZE, FLATTENED_IMAGE_SIZE
            ),
        )
        model_protonet = PrototypicalNetworks(nn.Flatten())

        model_feat.process_support_set(support_images, support_labels)
        model_protonet.process_support_set(support_images, support_labels)

        assert not model_feat.prototypes.isclose(
            model_protonet.prototypes, atol=1e-02
        ).all()
