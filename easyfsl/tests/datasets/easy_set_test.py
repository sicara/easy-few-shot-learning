import json
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from easyfsl.datasets import EasySet
from easyfsl.datasets.tiered_imagenet import TieredImageNet


def init_easy_set(specs):
    buffer = json.dumps(specs)
    with patch("builtins.open", mock_open(read_data=buffer)):
        with patch("pathlib.Path.glob") as mock_glob:
            mock_glob.return_value = [Path("a.jpeg"), Path("b.jpeg")]
            EasySet(Path("dummy.json"))


class TestEasySetInit:
    @staticmethod
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize(
        "specs",
        [
            {
                "class_names": ["class_1", "class_2"],
                "class_roots": ["path/to/class_1_folder", "path/to/class_2_folder"],
                "extra_key": [],
            },
        ],
    )
    def test_init_does_not_break_when_specs_are_ok(specs):
        init_easy_set(specs)

    @staticmethod
    @pytest.mark.parametrize(
        "specs_file_str",
        [
            "path/to/file.csv",
            "file.png",
        ],
    )
    def test_init_does_not_accept_non_json_specs(specs_file_str):
        with pytest.raises(ValueError):
            EasySet(Path(specs_file_str))

    @staticmethod
    @pytest.mark.parametrize(
        "specs",
        [
            {"class_roots": ["path/to/class_1_folder", "path/to/class_2_folder"]},
            {
                "class_names": ["class_1", "class_2"],
            },
            {
                "class_names": ["class_1", "class_2", "class_3"],
                "class_roots": ["path/to/class_1_folder", "path/to/class_2_folder"],
            },
            {
                "class_names": ["class_1", "class_2"],
                "class_roots": [
                    "path/to/class_1_folder",
                    "path/to/class_2_folder",
                    "path/to/class_3_folder",
                ],
            },
        ],
    )
    def test_init_returns_error_when_specs_dont_match_template(specs):
        with pytest.raises(ValueError):
            init_easy_set(specs)


class TestEasySetListDataInstances:
    @staticmethod
    @pytest.mark.parametrize(
        "class_roots,images,labels",
        [
            (
                [
                    "path/to/class_1_folder",
                    "path/to/class_2_folder",
                    "path/to/class_3_folder",
                ],
                [
                    "a.png",
                    "b.png",
                    "a.png",
                    "b.png",
                    "a.png",
                    "b.png",
                ],
                [0, 0, 1, 1, 2, 2],
            )
        ],
    )
    def test_list_data_instances_returns_expected_values(
        class_roots, images, labels, mocker
    ):
        mocker.patch("pathlib.Path.glob", return_value=[Path("a.png"), Path("b.png")])
        mocker.patch("pathlib.Path.is_file", return_value=True)

        assert (images, labels) == EasySet.list_data_instances(class_roots)

    @staticmethod
    @pytest.mark.parametrize(
        "images, all_files",
        [
            (
                [
                    # These must be sorted
                    "a.bmp",
                    "a.jpeg",
                    "a.jpg",
                    "a.png",
                ],
                [
                    "a.png",
                    "a.jpg",
                    "a.txt",
                    "a.bmp",
                    "a.jpeg",
                    "a.tmp",
                ],
            ),
        ],
    )
    def test_list_data_instances_lists_only_images(images, all_files, mocker):
        mocker.patch(
            "pathlib.Path.glob",
            return_value=[Path(file_name) for file_name in all_files],
        )
        mocker.patch("pathlib.Path.is_file", return_value=True)

        assert images == EasySet.list_data_instances(["abc"])[0]


class TestTieredImagenet:
    @staticmethod
    def test_tiered_imagenet_raises_error_if_wrong_split():
        with pytest.raises(ValueError):
            TieredImageNet("nope")

    @staticmethod
    @pytest.mark.parametrize(
        "split",
        [
            "train",
            "val",
            "test",
        ],
    )
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_tiered_imagenet_builds_easyset(split, mocker):
        mocker.patch(
            "pathlib.Path.glob",
            return_value=[Path("a.png"), Path("b.png")],
        )
        dataset = TieredImageNet(split)
        assert isinstance(dataset, EasySet)
