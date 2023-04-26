import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from numpy import ndarray
from torch import Tensor

from .few_shot_dataset import FewShotDataset


class FeaturesDataset(FewShotDataset):
    def __init__(
        self,
        labels: List[int],
        embeddings: Tensor,
        class_names: List[str],
    ):
        """
        Initialize a FeaturesDataset from explicit labels, class_names and embeddings.
        You can also initialize a FeaturesDataset from:
            - a dataframe with from_dataframe();
            - a parquet file with from_dataframe_parquet();
            - a dictionary with from_dict();
            - a pickle file with from_dict_pickle().
        Args:
            labels: list of labels, one for each embedding
            embeddings: tensor of embeddings with shape (n_images_for_this_class, **embedding_dimension)
            class_names: the name of the class associated to each integer label
                (length is the number of unique integers in labels)
        """
        self.labels = labels
        self.embeddings = embeddings
        self.class_names = class_names

    @classmethod
    def from_dataframe(cls, source_dataframe: pd.DataFrame):
        """
        Instantiate a FeaturesDataset from a dataframe.
        embeddings and class_names are directly inferred from the dataframe's content,
        while labels are inferred from the class_names.
        Args:
            source_dataframe: must have the columns embedding and class_name
        """
        if not {"embedding", "class_name"}.issubset(source_dataframe.columns):
            raise ValueError(
                f"Source dataframe must have the columns embedding and class_name, "
                f"but has columns {source_dataframe.columns}"
            )

        class_names = list(source_dataframe.class_name.unique())
        labels = list(
            source_dataframe.class_name.map(
                {
                    class_name: class_id
                    for class_id, class_name in enumerate(class_names)
                }
            )
        )
        embeddings = torch.tensor(list(source_dataframe.embedding))
        return cls(labels, embeddings, class_names)

    @classmethod
    def from_dataframe_parquet(cls, source_dataframe_parquet: Union[str, Path]):
        """
        Instantiate a FeaturesDataset from a parquet file containing a dataframe,
        using the from_dataframe() method.
        Args:
            source_dataframe_parquet: path to a parquet file containing the columns embedding and class_name
        """
        return cls.from_dataframe(pd.read_parquet(source_dataframe_parquet))

    @classmethod
    def from_dict(cls, source_dict: Dict[str, Union[ndarray, Tensor]]):
        """
        Instantiate a FeaturesDataset from a dictionary.
        Args:
            source_dict: each key is a class's name and each value is an array-like
                with shape (n_images_for_this_class, **embedding_dimension)
        """
        class_names = []
        labels = []
        embeddings_list = []
        for class_id, (class_name, class_embeddings) in enumerate(source_dict.items()):
            class_names.append(class_name)
            embeddings_list.append(torch.tensor(class_embeddings))
            labels += len(class_embeddings) * [class_id]
        return cls(labels, torch.cat(embeddings_list), class_names)

    @classmethod
    def from_dict_pickle(cls, source_dict_pickle: Union[str, Path]):
        """
        Instantiate a FeaturesDataset from a pickle file containing a dictionary
        using the from_dict() method.
        Args:
            source_dict_pickle: path to a pickle file containing a dictionary
                with the same format as the one expected by from_dict
        """
        with open(source_dict_pickle, "rb") as file:
            return cls.from_dict(json.load(file))

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        return self.embeddings[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)

    def get_labels(self) -> List[int]:
        return self.labels

    def number_of_classes(self):
        return len(self.class_names)
