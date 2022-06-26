from pathlib import Path

import pytest
from torchvision import transforms

from easyfsl.datasets import SupportSetFolder
from easyfsl.datasets.support_set_folder import NOT_A_TENSOR_ERROR_MESSAGE


class TestSupportSetFolderInit:
    @staticmethod
    @pytest.mark.parametrize(
        "root",
        [
            Path("easyfsl/tests/datasets/resources/balanced_support_set"),
            Path("easyfsl/tests/datasets/resources/unbalanced_support_set"),
        ],
    )
    def test_init_does_not_break_when_support_set_is_ok_and_not_custom_args(root):
        SupportSetFolder(root)

    @staticmethod
    @pytest.mark.parametrize(
        "root,transform",
        [
            (
                Path("easyfsl/tests/datasets/resources/balanced_support_set"),
                transforms.Resize((10, 10)),
            ),
        ],
    )
    def test_init_raises_type_error_when_transform_does_not_input_tensor(
        root, transform
    ):
        with pytest.raises(TypeError) as exc_info:
            SupportSetFolder(root, transform=transform)
        assert exc_info.value.args[0] == NOT_A_TENSOR_ERROR_MESSAGE

    @staticmethod
    @pytest.mark.parametrize(
        "root",
        [
            Path("easyfsl/tests/datasets/resources/empty_support_set"),
        ],
    )
    def test_init_raises_error_when_support_set_is_empty(root):
        with pytest.raises(FileNotFoundError):
            SupportSetFolder(root)
