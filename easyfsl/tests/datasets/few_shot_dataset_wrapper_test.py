import pytest
from torchvision.datasets import FakeData, ImageFolder

from easyfsl.datasets import FewShotDatasetWrapper


class TestInit:
    @staticmethod
    @pytest.mark.parametrize(
        "source_dataset,expected_labels",
        [
            (
                ImageFolder("easyfsl/tests/datasets/resources/balanced_support_set"),
                [0, 0, 1, 1, 2, 2],
            ),
            (
                ImageFolder("easyfsl/tests/datasets/resources/unbalanced_support_set"),
                [0, 0, 1, 2, 2, 2, 2, 2],
            ),
        ],
    )
    def test_default_init_retrieves_correct_labels(source_dataset, expected_labels):
        wrapped_dataset = FewShotDatasetWrapper(source_dataset)
        assert wrapped_dataset.get_labels() == expected_labels

    @staticmethod
    @pytest.mark.parametrize(
        "image_position_in_get_item_output,label_position_in_get_item_output",
        [
            (1, 0),
            (1, 2),
            (4, 5),
            (0, 10),
            (10, 0),
        ],
    )
    def test_default_init_retrieves_correct_labels(
        image_position_in_get_item_output,
        label_position_in_get_item_output,
    ):
        class FakeImageFolder(ImageFolder):
            def __getitem__(self, item):
                image, label = super().__getitem__(item)
                output_as_list = [None] * (
                    max(
                        image_position_in_get_item_output,
                        label_position_in_get_item_output,
                    )
                    + 1
                )
                output_as_list[image_position_in_get_item_output] = image
                output_as_list[label_position_in_get_item_output] = label
                return tuple(output_as_list)

        source_dataset = FakeImageFolder(
            "easyfsl/tests/datasets/resources/unbalanced_support_set"
        )
        wrapped_dataset = FewShotDatasetWrapper(
            source_dataset,
            image_position_in_get_item_output,
            label_position_in_get_item_output,
        )
        assert wrapped_dataset.get_labels() == [0, 0, 1, 2, 2, 2, 2, 2]

    @staticmethod
    def test_raises_error_when_input_positions_are_out_of_item_range():
        pass
