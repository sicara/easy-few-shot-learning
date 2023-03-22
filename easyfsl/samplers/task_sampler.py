import random
from typing import List, Tuple, Iterator

import torch
from torch import Tensor
from torch.utils.data import Sampler
from typing import Union
from easyfsl.datasets import FewShotDataset


class TaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(
        self,
        dataset: FewShotDataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int,
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks

        self.items_per_label = {}
        for item, label in enumerate(dataset.get_labels()):
            if label in self.items_per_label:
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_tasks):
            yield torch.cat(
                [
                    # pylint: disable=not-callable
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], self.n_shot + self.n_query
                        )
                    )
                    # pylint: enable=not-callable
                    for label in random.sample(self.items_per_label.keys(), self.n_way)
                ]
            ).tolist()

    def episodic_collate_fn(
        self, input_data: List[Tuple[Tensor, Union[Tensor, int]]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[int]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image as an int or a 0-dim tensor
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images,
                - their labels,
                - query images,
                - their labels,
                - the dataset class ids of the class sampled in the episode
        Raises:
            TypeError : Wrong type of input 
        """
        #checking the input type
        right_input_type, o_dim_tesors_labesl= self.check_episodic_collate_fn_input(input_data)
        if not right_input_type:
            raise TypeError(
                "Illegal type of input."
                "check out the type of the output of the .getitem() method of your dataset and make sure it's" 
                "a List[Tuple[Tensor, int]] or List[Tuple[Tensor, 0-dim Tensor]]."
                )
        #if the input is List[Tuple[Tensor, 0-dim Tensor]], turn the tensor into an int
        if  o_dim_tesors_labesl:
            self.o_tesors_to_ints(input_data)
            
        true_class_ids = list({x[1] for x in input_data})

        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        # pylint: disable=not-callable
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        # pylint: enable=not-callable

        support_images = all_images[:, : self.n_shot].reshape(
            (-1, *all_images.shape[2:])
        )
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )
        
    def check_episodic_collate_fn_input(
        self, input_data: List[Tuple[Tensor, Union[Tensor, int]]]
    ) -> bool:
        """
        Check the type of the input for the episodic_collate_fn method.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image as an int or a 0-dim tensor
        Returns:
            bool:
                - True if the input is of correct type,
                - False if input is of wrong type.
        """
        #that flag is true when the input labels are  0-dim tensors
        o_dim_tesors_labesl=False
        for img, label in input_data:
            #if that label isn't int then check if it's a 0-tensor
            if not  (isinstance(img,torch.Tensor) and isinstance(label,int)):
                if not (isinstance(img,torch.Tensor) and isinstance(label,torch.Tensor) and label.ndim==0):
                    return False,o_dim_tesors_labesl
                else:
                    o_dim_tesors_labesl=True
        #if that loop went fine, then the data is the correct type
        return True,o_dim_tesors_labesl
    
    def o_tesors_to_ints(
        self, input_data: List[Tuple[Tensor, Union[Tensor, int]]]
    ) -> None:
        """
        Turn 0-dim tensors into ints.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image as a 0-dim tensor
        """ 
        for idx,_ in enumerate(input_data):
            input_data[idx]=(input_data[idx][0],int(input_data[idx][1]))
        
        
        
