from pathlib import Path

from easyfsl.datasets import EasySet, FewShotDataset

CUB_SPECS_DIR = Path("data/CUB")


# pylint: disable=invalid-name
def CUB(split: str, **kwargs) -> FewShotDataset:
    """
    Build the CUB dataset for the specific split.
    Args:
        split: one of the available split (typically train, val, test).

    Returns:
        the constructed dataset using EasySet

    Raises:
        ValueError: if the specified split cannot be associated with a JSON spec file
            from CUB's specs directory
    """
    specs_file = CUB_SPECS_DIR / f"{split}.json"
    if specs_file.is_file():
        return EasySet(specs_file=specs_file, **kwargs)

    raise ValueError(f"Could not find specs file {specs_file.name} in {CUB_SPECS_DIR}")


# pylint: enable=invalid-name
