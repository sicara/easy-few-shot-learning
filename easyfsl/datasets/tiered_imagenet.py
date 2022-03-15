from pathlib import Path

from easyfsl.datasets import EasySet, FewShotDataset

TIERED_IMAGENET_SPECS_DIR = Path("data/tiered_imagenet")


# pylint: disable=invalid-name
def TieredImageNet(split: str, **kwargs) -> FewShotDataset:
    """
    Build the tieredImageNet dataset for the specific split.
    Args:
        split: one of the available split (typically train, val, test).

    Returns:
        the constructed dataset using EasySet

    Raises:
        ValueError: if the specified split cannot be associated with a JSON spec file
            from tieredImageNet's specs directory
    """
    specs_file = TIERED_IMAGENET_SPECS_DIR / f"{split}.json"
    if specs_file.is_file():
        return EasySet(specs_file=specs_file, **kwargs)

    raise ValueError(
        f"Could not find specs file {specs_file.name} in {TIERED_IMAGENET_SPECS_DIR}"
    )


# pylint: enable=invalid-name
