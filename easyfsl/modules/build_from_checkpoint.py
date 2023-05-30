from pathlib import Path

import torch

from easyfsl.methods.utils import strip_prefix
from easyfsl.modules.feat_resnet12 import FEATResNet12, feat_resnet12


def feat_resnet12_from_checkpoint(
    checkpoint_path: Path, device: str, **kwargs
) -> FEATResNet12:
    model = feat_resnet12(**kwargs).to(device)

    state_dict = torch.load(str(checkpoint_path), map_location=device)["params"]

    backbone_missing_keys, _ = model.load_state_dict(
        strip_prefix(state_dict, "encoder."), strict=False
    )

    if len(backbone_missing_keys) > 0:
        raise ValueError(f"Missing keys for backbone: {backbone_missing_keys}")

    return model
