import pytest
from torchvision.models import resnet18

from easyfsl.methods import FewShotClassifier


class TestFSCAbstractMethods:
    @staticmethod
    def test_forward_raises_error_when_not_implemented():
        with pytest.raises(NotImplementedError):
            model = FewShotClassifier(resnet18())
            model(None)

    @staticmethod
    def test_process_support_set_raises_error_when_not_implemented():
        with pytest.raises(NotImplementedError):
            model = FewShotClassifier(resnet18())
            model.process_support_set(None, None)
