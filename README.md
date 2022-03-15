# Easy Few-Shot Learning
![Python Versions](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-%23EBBD68.svg)
![CircleCI](https://img.shields.io/circleci/build/github/sicara/easy-few-shot-learning)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sicara/easy-few-shot-learning/blob/master/notebooks/my_first_few_shot_classifier.ipynb)

Ready-to-use code and tutorial notebooks to boost your way into few-shot image classification. 
This repository is made for you if:

- you're new to few-shot learning and want to learn;
- or you're looking for reliable, clear and easily usable code that you can use for your projects.

Don't get lost in large repositories with hundreds of methods and no explanation on how to use them. Here, we want each line
of code to be covered by a tutorial.
## What's in there?

### Notebooks: learn and practice
You want to learn few-shot learning and don't know where to start? Start with our tutorial.

- **[First steps into few-shot image classification](notebooks/my_first_few_shot_classifier.ipynb)**: 
basically Few-Shot Learning 101, in less than 15mn.

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sicara/easy-few-shot-learning/blob/master/notebooks/my_first_few_shot_classifier.ipynb)

- **[Example of episodic training](notebooks/episodic_training.ipynb)**: 
use it as a starting point if you want to design a script for episodic training using EasyFSL.

### Code that you can use and understand

**State-Of-The-Art Few-Shot Learning methods:**

- [FewShotClassifier](easyfsl/methods/few_shot_classifier.py): an abstract class with methods that can be used for 
  any few-shot classification algorithm
- [Prototypical Networks](easyfsl/methods/prototypical_networks.py)
- [Matching Networks](easyfsl/methods/matching_networks.py)
- [Relation Networks](easyfsl/methods/relation_networks.py)
- [Fine-Tune](easyfsl/methods/finetune.py)
- [BD-CSPN](easyfsl/methods/bd_cspn.py)
- [Transductive Fine-Tuning](easyfsl/methods/transductive_finetuning.py)
- [Transductive Information Maximization](easyfsl/methods/tim.py)

To reproduce their results, you can use the [standard network architectures](easyfsl/modules/predesigned_modules.py) 
used in Few-Shot Learning research. They're also a feature of EasyFSL!

**Tools for data loading:**

Data loading in FSL is a bit different from standard classification because we sample batches of
instances in the shape of few-shot classification tasks. No sweat! In EasyFSL you have:

- [TaskSampler](easyfsl/samplers/task_sampler.py): an extension of the standard PyTorch Sampler object, to sample batches in the shape of few-shot classification tasks
- [FewShotDataset](easyfsl/datasets/few_shot_dataset.py): an abstract class to standardize the interface of any dataset you'd like to use
- [EasySet](easyfsl/datasets/easy_set.py): a ready-to-use FewShotDataset object to handle datasets of images with a class-wise directory split

**And also:** [some utilities](easyfsl/utils.py) that I felt I often used in my research, so I'm sharing with you.

### Datasets to test your model

There are enough datasets used in Few-Shot Learning for anyone to get lost in them. They're all here, 
explicited, downloadable and easy-to-use, in EasyFSL. 

**[CU-Birds](http://www.vision.caltech.edu/visipedia/CUB-200.html)**

We provide [a script](scripts/download_CUB.sh) to download and extract the dataset, 
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

## [Deprecated] QuickStart


1. Install the package with pip: 
   
```pip install easyfsl```

Note: alternatively, you can clone the repository so that you can modify the code as you wish.
   
2. [Download your data ](#datasets-to-test-your-model) (for instance CUB).

4. From the training subset of CUB, create a dataloader that yields few-shot classification tasks:

```python
from easyfsl.datasets import CUB
from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader

train_set = CUB(split="train", training=True)
train_sampler = TaskSampler(
    train_set, n_way=5, n_shot=5, n_query=10, n_tasks=40000
)
train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)
```

5. Create and train a model

```python
from easyfsl.methods import PrototypicalNetworks
from torch import nn
from torch.optim import Adam
from torchvision.models import resnet18

convolutional_network = resnet18(pretrained=False)
convolutional_network.fc = nn.Flatten()
model = PrototypicalNetworks(convolutional_network).cuda()

optimizer = Adam(params=model.parameters())

model.fit(train_loader, optimizer)
```
   **Note:** you can also define a validation data loader and use as an additional argument to `fit`
in order to use validation during your training.

   **Troubleshooting:** a ResNet18 with a batch size of (5 * (5+10)) = 75 would use about 4.2GB on your GPU.
If you don't have it, switch to CPU, choose a smaller model or reduce the batch size (in `TaskSampler` above).

6. Evaluate your model on the test set

```python
test_set = EasySet(specs_file="./data/CUB/test.json", training=False)
test_sampler = TaskSampler(
    test_set, n_way=5, n_shot=5, n_query=10, n_tasks=100
)
test_loader = DataLoader(
    test_set,
    batch_sampler=test_sampler,
    num_workers=12,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)

accuracy = model.evaluate(test_loader)
print(f"Average accuracy : {(100 * accuracy):.2f}")
```

## Contribute
This project is very open to contributions! You can help in various ways:
- raise issues
- resolve issues already opened
- tackle new features from the roadmap
- fix typos, improve code quality



