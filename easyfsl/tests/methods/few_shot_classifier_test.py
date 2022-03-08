import pytest

from easyfsl.methods import FewShotClassifier
from easyfsl.modules.predesigned_modules import resnet12


class TestFSCAbstractMethods:
    @staticmethod
    def test_forward_raises_error_when_not_implemented():
        with pytest.raises(NotImplementedError):
            model = FewShotClassifier(resnet12())
            model(None)

    @staticmethod
    def test_process_support_set_raises_error_when_not_implemented():
        with pytest.raises(NotImplementedError):
            model = FewShotClassifier(resnet12())
            model.process_support_set(None, None)
