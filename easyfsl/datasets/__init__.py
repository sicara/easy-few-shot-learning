# FewShotDataset and EasySet must be imported before their child classes
from .few_shot_dataset import FewShotDataset  # isort:skip
from .easy_set import EasySet  # isort:skip
from .cub import CUB
from .danish_fungi import DanishFungi
from .mini_imagenet import MiniImageNet
from .support_set_folder import SupportSetFolder
from .tiered_imagenet import TieredImageNet
