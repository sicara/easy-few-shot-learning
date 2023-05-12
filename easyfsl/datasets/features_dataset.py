import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
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
            - a dictionary with from_dict();
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
            source_dataframe: must have the columns embedding and class_name.
                Embeddings must be tensors or numpy arrays.
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
        if len(source_dataframe) == 0:
            warnings.warn(
                UserWarning(
                    "Empty source dataframe. Initializing an empty FeaturesDataset."
                )
            )
            embeddings = torch.empty(0)
        else:
            embeddings = torch.from_numpy(np.stack(list(source_dataframe.embedding)))

        return cls(labels, embeddings, class_names)

    @classmethod
    def from_dict(cls, source_dict: Dict[str, Union[ndarray, Tensor]]):
        """
        Instantiate a FeaturesDataset from a dictionary.
        Args:
            source_dict: each key is a class's name and each value is a numpy array or torch tensor
                with shape (n_images_for_this_class, **embedding_dimension)
        """
        class_names = []
        labels = []
        embeddings_list = []
        for class_id, (class_name, class_embeddings) in enumerate(source_dict.items()):
            class_names.append(class_name)
            if isinstance(class_embeddings, ndarray):
                embeddings_list.append(torch.from_numpy(class_embeddings))
            elif isinstance(class_embeddings, Tensor):
                embeddings_list.append(class_embeddings)
            else:
                raise ValueError(
                    f"Each value of the source_dict must be a ndarray or torch tensor, "
                    f"but the value for class {class_name} is {class_embeddings}"
                )
            labels += len(class_embeddings) * [class_id]
        return cls(labels, torch.cat(embeddings_list), class_names)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        return self.embeddings[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)

    def get_labels(self) -> List[int]:
        return self.labels

    def number_of_classes(self):
        return len(self.class_names)
