# Easy Few-Shot Learning
Ready-to-use code and tutorial notebooks to boost your way into few-shot image classification. 
This repository is made for you if:

- you're new to few-shot learning and want to learn;
- or you're looking for reliable, clear and easily usable code that you can use for your projects.

Don't get lost in large repositories with hundreds of methods and no explanation on how to use them. Here, we want each line
of code to be covered by a tutorial.
## What's in there?

### Notebooks: learn and practice
You want to learn few-shot learning and don't know where to start? Start with our tutorials.

- **[First steps into few-shot image classification](notebooks/my_first_few_shot_classifier.ipynb)** 
  
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sicara/easy-few-shot-learning/blob/master/notebooks/my_first_few_shot_classifier.ipynb)

### Code that you can use and understand

**Models:**

- [AbstractMetaLearner](easyfsl/methods/abstract_meta_learner.py): an abstract class with methods that can be used for 
  any meta-trainable algorithm
  
- [Prototypical Networks](easyfsl/methods/prototypical_networks.py)

**Helpers:**

- [EasySet](easyfsl/data_tools/easy_set.py): a ready-to-use Dataset object to handle datasets of images with a class-wise directory split
- [TaskSampler](easyfsl/data_tools/tasl_sampler.py): samples batches in the shape of few-shot classification tasks

### Datasets to test your model

- [CU-Birds](http://www.vision.caltech.edu/visipedia/CUB-200.html): we provide [a script](scripts/download_CUB.sh) to download
and extract the dataset, along with [a meta-train/meta-val/meta-test split](data/CUB) along classes. The dataset is
  ready-to-use with [EasySet](easyfsl/data_tools/easy_set.py).

## 3mn QuickStart
Work in progress

## Roadmap

- [ ] Implement unit tests
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



