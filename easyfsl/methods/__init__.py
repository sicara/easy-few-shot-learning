# FewShotClassifier must be imported before its child classes
from .few_shot_classifier import FewShotClassifier

from .bd_cspn import BDCSPN
from .finetune import Finetune
from .matching_networks import MatchingNetworks
from .prototypical_networks import PrototypicalNetworks
from .relation_networks import RelationNetworks
from .tim import TIM
from .transductive_finetuning import TransductiveFinetuning
