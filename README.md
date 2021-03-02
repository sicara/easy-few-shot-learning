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

- **[First steps into few-shot image classification](notebooks/my_first_few_shot_classifier.ipynb)** 
  
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sicara/easy-few-shot-learning/blob/master/notebooks/my_first_few_shot_classifier.ipynb)

### Code that you can use and understand

**Models:**

- [AbstractMetaLearner](easyfsl/methods/abstract_meta_learner.py): an abstract class with methods that can be used for 
  any meta-trainable algorithm
  
- [Prototypical Networks](easyfsl/methods/prototypical_networks.py)

**Tools for data loading:**

- [EasySet](easyfsl/data_tools/easy_set.py): a ready-to-use Dataset object to handle datasets of images with a class-wise directory split
- [TaskSampler](easyfsl/data_tools/task_sampler.py): samples batches in the shape of few-shot classification tasks

### Datasets to test your model

- [CU-Birds](http://www.vision.caltech.edu/visipedia/CUB-200.html): we provide [a script](scripts/download_CUB.sh) to download
and extract the dataset, along with [a meta-train/meta-val/meta-test split](data/CUB) along classes. The dataset is
  ready-to-use with [EasySet](easyfsl/data_tools/easy_set.py).

## QuickStart
1. Install the package with pip: 
   
```pip install git+https://github.com/sicara/easy-few-shot-learning.git```

Note: alternatively, you can clone the repository so that you can modify the code as you wish.
   
2. Download [CU-Birds](http://www.vision.caltech.edu/visipedia/CUB-200.html) and the few-shot train/val/test split:

```
mkdir -p data/CUB && cd data/CUB
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GDr1OkoXdhaXWGA8S3MAq3a522Tak-nx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GDr1OkoXdhaXWGA8S3MAq3a522Tak-nx" -O images.tgz
rm -rf /tmp/cookies.txt
tar  --exclude='._*' -zxvf images.tgz
wget https://raw.githubusercontent.com/sicara/easy-few-shot-learning/master/data/CUB/train.json
wget https://raw.githubusercontent.com/sicara/easy-few-shot-learning/master/data/CUB/val.json
wget https://raw.githubusercontent.com/sicara/easy-few-shot-learning/master/data/CUB/test.json
cd ...
```
   
3. Check that you have a 680,9MB `images` folder in `./data/CUB` along with three JSON files.

4. From the training subset of CUB, create a dataloader that yields few-shot classification tasks:
```python
from easyfsl.data_tools import EasySet, TaskSampler
from torch.utils.data import DataLoader

train_set = EasySet(specs_file="./data/CUB/train.json", training=True)
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

   **Troubleshooting:** a ResNet18 with a batch size of (5 * (5+10)) = 75 whould use about 4.2GB on your GPU.
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

model.evaluate(test_loader)
```

## Roadmap

- [ ] Implement unit tests
- [ ] Add validation to `AbstractMetaLearner.fit()`
- [ ] Integrate more methods: 
  - [ ] Matching Networks
  - [ ] Relation Networks
  - [ ] MAML
  - [ ] Transductive Propagation Network
- [ ] Integrate non-episodic training
- [ ] Integrate more benchmarks:
  - [ ] miniImageNet
  - [ ] tieredImageNet
  - [ ] Meta-Dataset

## Contribute
This project is very open to contributions! You can help in various ways:
- raise issues
- resolve issues already opened
- tackle new features from the roadmap
- fix typos, improve code quality



