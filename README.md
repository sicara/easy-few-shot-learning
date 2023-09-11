# Easy Few-Shot Learning
![Python Versions](https://img.shields.io/pypi/pyversions/easyfsl?logo=python&logoColor=white&style=for-the-badge)
![License: MIT](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![CircleCI](https://img.shields.io/circleci/build/github/sicara/easy-few-shot-learning?logo=circleci&style=for-the-badge)
![PyPi Downloads](https://img.shields.io/pypi/dm/easyfsl?logo=pypi&logoColor=white&style=for-the-badge)
![Last Release](https://img.shields.io/github/release-date/sicara/easy-few-shot-learning?label=Last%20Release&logo=pypi&logoColor=white&style=for-the-badge)
![Github Issues](https://img.shields.io/github/issues-closed/sicara/easy-few-shot-learning?color=green&logo=github&style=for-the-badge)

Ready-to-use code and tutorial notebooks to boost your way into few-shot image classification. 
This repository is made for you if:

- you're new to few-shot learning and want to learn;
- or you're looking for reliable, clear and easily usable code that you can use for your projects.

Don't get lost in large repositories with hundreds of methods and no explanation on how to use them. Here, we want each line
of code to be covered by a tutorial.
## What's in there?

### Notebooks: learn and practice
You want to learn few-shot learning and don't know where to start? Start with our tutorials.

| Notebook                                                                                       | Description                                                                                                                                                                                  | Colab                                                                                                                                                                                                              |
|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [First steps into few-shot image classification](notebooks/my_first_few_shot_classifier.ipynb) | Basically Few-Shot Learning 101, in less than 15min.                                                                                                                                         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sicara/easy-few-shot-learning/blob/master/notebooks/my_first_few_shot_classifier.ipynb)      |
| [Example of episodic training](notebooks/episodic_training.ipynb)                              | Use it as a starting point if you want to design a script for episodic training using EasyFSL.                                                                                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sicara/easy-few-shot-learning/blob/master/notebooks/episodic_training.ipynb)                 |
| [Example of classical training](notebooks/classical_training.ipynb)                            | Use it as a starting point if you want to design a script for classical training using EasyFSL.                                                                                              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sicara/easy-few-shot-learning/blob/master/notebooks/classical_training.ipynb)                |
| [Test with pre-extracted embeddings](notebooks/inference_with_extracted_embeddings.ipynb)      | Most few-shot methods use a frozen backbone at test-time. With EasyFSL, you can extract all embeddings for your dataset once and for all, and then perform inference directly on embeddings. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sicara/easy-few-shot-learning/blob/master/notebooks/inference_with_extracted_embeddings.ipynb) |

### Code that you can use and understand

**State-Of-The-Art Few-Shot Learning methods:**

With 11 built-in methods, EasyFSL is the most comprehensive open-source Few-Shot Learning library!

- [Prototypical Networks](easyfsl/methods/prototypical_networks.py)
- [SimpleShot](easyfsl/methods/simple_shot.py)
- [Matching Networks](easyfsl/methods/matching_networks.py)
- [Relation Networks](easyfsl/methods/relation_networks.py)
- [FEAT](easyfsl/methods/feat.py)
- [Fine-Tune](easyfsl/methods/finetune.py)
- [BD-CSPN](easyfsl/methods/bd_cspn.py)
- [LaplacianShot](easyfsl/methods/laplacian_shot.py)
- [Transductive Information Maximization](easyfsl/methods/tim.py)
- [PT-MAP](easyfsl/methods/pt_map.py)
- [Transductive Fine-Tuning](easyfsl/methods/transductive_finetuning.py)

We also provide a [FewShotClassifier](easyfsl/methods/few_shot_classifier.py) class to quickstart your implementation 
of any few-shot classification algorithm, as well as [commonly used architectures](easyfsl/modules).

See the benchmarks section below for more details on the methods.

**Tools for data loading:**

Data loading in FSL is a bit different from standard classification because we sample batches of
instances in the shape of few-shot classification tasks. No sweat! In EasyFSL you have:

- [TaskSampler](easyfsl/samplers/task_sampler.py): an extension of the standard PyTorch Sampler object, to sample batches in the shape of few-shot classification tasks
- [FewShotDataset](easyfsl/datasets/few_shot_dataset.py): an abstract class to standardize the interface of any dataset you'd like to use
- [EasySet](easyfsl/datasets/easy_set.py): a ready-to-use FewShotDataset object to handle datasets of images with a class-wise directory split
- [WrapFewShotDataset](easyfsl/datasets/wrap_few_shot_dataset.py): a wrapper to transform any dataset into a FewShotDataset object
- [FeaturesDataset](easyfsl/datasets/features_dataset.py): a dataset to handle pre-extracted features
- [SupportSetFolder](easyfsl/datasets/support_set_folder.py): a dataset to handle support sets stored in a directory

**Scripts to reproduce our benchmarks:**

- `scripts/predict_embeddings.py` to extract all embeddings from a dataset with a given pre-trained backbone
- `scripts/benchmark_methods.py` to evaluate a method on a test dataset using pre-extracted embeddings.

**And also:** [some utilities](easyfsl/utils.py) that I felt I often used in my research, so I'm sharing with you.

### Datasets to test your model

There are enough datasets used in Few-Shot Learning for anyone to get lost in them. They're all here, 
explicited, downloadable and easy-to-use, in EasyFSL. 

**[CU-Birds](http://www.vision.caltech.edu/visipedia/CUB-200.html)**

We provide a `make download-cub` recipe to download and extract the dataset, 
along with the standard [(train / val / test) split](data/CUB) along classes. 
Once you've downloaded the dataset, you can instantiate the Dataset objects in your code
with this super complicated process:

```python
from easyfsl.datasets import CUB

train_set = CUB(split="train", training=True)
test_set = CUB(split="test", training=False)
```

**[tieredImageNet](https://paperswithcode.com/dataset/tieredimagenet)**

To use it, you need the [ILSVRC2015](https://image-net.org/challenges/LSVRC/index.php) dataset. Once you have 
downloaded and extracted the dataset, ensure that its localisation on disk is consistent with the class paths
specified in the [specification files](data/tiered_imagenet). Then:

```python
from easyfsl.datasets import TieredImageNet

train_set = TieredImageNet(split="train", training=True)
test_set = TieredImageNet(split="test", training=False)
```

**[miniImageNet](https://paperswithcode.com/dataset/miniimagenet)**

Same as tieredImageNet, we provide the [specification files](data/mini_imagenet), 
but you need the [ILSVRC2015](https://image-net.org/challenges/LSVRC/index.php) dataset.
Once you have it:

```python
from easyfsl.datasets import MiniImageNet

train_set = MiniImageNet(root="where/imagenet/is", split="train", training=True)
test_set = MiniImageNet(root="where/imagenet/is", split="test", training=False)
```

Since miniImageNet is relatively small, you can also load it on RAM directly at instantiation simply by
adding `load_on_ram=True` to the constructor. 
It takes a few minutes but it can make your training significantly faster!

**[Danish Fungi](https://paperswithcode.com/paper/danish-fungi-2020-not-just-another-image)**

I've recently started using it as a Few-Shot Learning benchmarks, and I can tell you it's a great
playing field. To use it, first download the data:

```shell
# Download the original dataset (/!\ 110GB)
wget http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20-train_val.tar.gz
# Or alternatively the images reduced to 300px (6.5Gb)
wget http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20-300px.tar.gz
# And finally download the metadata (83Mb) to data/fungi/
wget https://public-sicara.s3.eu-central-1.amazonaws.com/easy-fsl/DF20_metadata.csv  -O data/fungi/DF20_metadata.csv
```

And then instantiate the dataset with the same process as always:

```python
from easyfsl.datasets import DanishFungi

dataset = DanishFungi(root="where/fungi/is")
```

Note that I didn't specify a train and test set because the CSV I gave you describes the whole dataset.
I recommend to use it to test models with weights trained on an other dataset (like ImageNet).
But if you want to propose a train/val/test split along classes, you're welcome to contribute!

## QuickStart


1. Install the package: ```pip install easyfsl``` or simply fork the repository.
   
2. [Download your data](#datasets-to-test-your-model).

3. Design your training and evaluation scripts. You can use our example notebooks for 
[episodic training](notebooks/episodic_training.ipynb) 
or [classical training](notebooks/classical_training.ipynb).

## Contribute
This project is very open to contributions! You can help in various ways:
- raise issues
- resolve issues already opened
- tackle new features from the roadmap
- fix typos, improve code quality

## Benchmarks

We used EasyFSL to benchmark a dozen methods. 
Inference times are computed over 1000 tasks using pre-extracted features. They are only indicative.
Note that the inference time for fine-tuning methods highly depends on the number of fine-tuning steps.

All methods hyperparameters are defined in [this JSON file](scripts/backbones_configs.json). 
They were selected on miniImageNet validation set. 
The procedure can be reproduced with `make hyperparameter-search`.
We decided to use miniImageNet's hyperparameters for all benchmarks in order to highlight the adaptability of
the different methods.
Note that all methods use L2 normalization of features, except for FEAT as it harms its performance.

There are no results for Mathing and Relation Networks as the trained weights for their additional modules are unavailable.

### miniImageNet & tieredImageNet

All methods use the same backbone: [a custom ResNet12](easyfsl/modules/feat_resnet12.py) using the trained parameters
provided by the authors from [FEAT](https://github.com/Sha-Lab/FEAT) 
(download: [miniImageNet](https://drive.google.com/file/d/1ixqw1l9XVxl3lh1m5VXkctw6JssahGbQ/view),
[tieredImageNet](https://drive.google.com/file/d/1M93jdOjAn8IihICPKJg8Mb4B-eYDSZfE/view)).

Best inductive and best transductive results for each column are shown in bold.

| Method                                                                    | Ind / Trans  | *mini*Imagenet<br/>1-shot | *mini*Imagenet<br/>5-shot | *tiered*Imagenet<br/>1-shot | *tiered*Imagenet<br/>5-shot | Time    |
|---------------------------------------------------------------------------|--------------|---------------------------|---------------------------|-----------------------------|-----------------------------|---------|
| **[ProtoNet](easyfsl/methods/prototypical_networks.py)**                  | Inductive    | 63.6                      | 80.4                      | 60.2                        | 77.4                        | 6s      |
| **[SimpleShot](easyfsl/methods/simple_shot.py)**                          | Inductive    | 63.6                      | **80.5**                  | 60.2                        | 77.4                        | 6s      |
| **[MatchingNet](easyfsl/methods/matching_networks.py)**                   | Inductive    | -                         | -                         | -                           | -                           | -       |
| **[RelationNet](easyfsl/methods/relation_networks.py)**                   | Inductive    | -                         | -                         | -                           | -                           | -       |
| **[Finetune](easyfsl/methods/finetune.py)**                               | Inductive    | 63.3                      | **80.5**                  | 59.8                        | **77.5**                    | 1mn33s  |
| **[FEAT](easyfsl/methods/feat.py)**                                       | Inductive    | **64.7**                  | 80.1                      | **61.3**                    | 76.2                        | 3s      |
| **[BD-CSPN](easyfsl/methods/bd_cspn.py)**                                 | Transductive | 69.8                      | 82.2                      | 66.3                        | 79.1                        | 7s      |
| **[LaplacianShot](easyfsl/methods/laplacian_shot.py)**                    | Transductive | 69.8                      | 82.3                      | 66.2                        | 79.2                        | 9s      |
| **[PT-MAP](easyfsl/methods/pt_map.py)**                                   | Transductive | **76.1**                  | **84.2**                  | **71.7**                    | **80.7**                    | 39mn40s |
| **[TIM](easyfsl/methods/tim.py)**                                         | Transductive | 74.3                      | **84.2**                  | 70.7                        | **80.7**                    | 3mn05s  |
| **[Transductive Finetuning](easyfsl/methods/transductive_finetuning.py)** | Transductive | 63.0                      | 80.6                      | 59.1                        | 77.5                        | 30s     |

To reproduce:

1. Download the [*mini*ImageNet](https://drive.google.com/file/d/1ixqw1l9XVxl3lh1m5VXkctw6JssahGbQ/view) 
   and [tieredImageNet](https://drive.google.com/file/d/1M93jdOjAn8IihICPKJg8Mb4B-eYDSZfE/view) weights for ResNet12 
   and save them under `data/models/feat_resnet12_mini_imagenet.pth` (resp. `tiered`).
2. Extract all embeddings from the test sets of all datasets with `make extract-all-features-with-resnet12`.
3. Run the evaluation scripts with `make benchmark-mini-imagenet` (resp. `tiered`).