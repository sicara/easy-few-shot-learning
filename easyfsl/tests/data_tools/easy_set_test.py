import json
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from easyfsl.data_tools import EasySet


def init_easy_set(specs):
    buffer = json.dumps(specs)
    with patch("builtins.open", mock_open(read_data=buffer)):
        with patch("pathlib.Path.glob") as mock_glob:
            mock_glob.return_value = [Path("a"), Path("b")]
            EasySet(Path("dummy.json"))


class TestEasySetInit:
    @staticmethod
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
    def test_list_data_instances_returns_expected_values(class_roots, images, labels):
        with patch("pathlib.Path.glob") as mock_glob:
            mock_glob.return_value = [Path("a.png"), Path("b.png")]
            with patch("pathlib.Path.is_file") as mock_is_file:
                mock_is_file.return_value = True
                assert (images, labels) == EasySet.list_data_instances(class_roots)
