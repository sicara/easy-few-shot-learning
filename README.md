# tuto-fsl
Ready-to-use code and tutorial notebooks to boost your way into few-shot image classification.

## What's in there?

### Notebooks

- **[First steps into few-shot image classification](notebooks/my_first_few_shot_classifier.ipynb)** 
  
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sicara/tuto-fsl/blob/master/notebooks/my_first_few_shot_classifier.ipynb)

### Code

**Models:**

- [AbstractMetaLearner](src/abstract_meta_learner.py): an abstract class with methods that can be used for 
  any meta-trainable algorithm
  
- [Prototypical Networks](src/prototypical_networks.py)

**Helpers:**

- [EasySet](src/dataset.py): a ready-to-use Dataset object to handle datasets of images with a class-wise directory split
- [TaskSampler](src/sampler.py): samples batches in the shape of few-shot classification tasks

### Datasets

- [CU-Birds](http://www.vision.caltech.edu/visipedia/CUB-200.html): we provide [a script](scripts/download_CUB.sh) to download
and extract the dataset, along with [a meta-train/meta-val/meta-test split](data/CUB) along classes. The dataset is
  ready-to-use with [EasySet](src/dataset.py).
  
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



