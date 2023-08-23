import pytest
import torch

from easyfsl.methods import FewShotClassifier
from easyfsl.modules.predesigned_modules import resnet12


class TestFSCAbstractMethods:
    @staticmethod
    def test_forward_raises_error_when_not_implemented():
        with pytest.raises(NotImplementedError):
            model = FewShotClassifier(resnet12())
            model(None)


class TestFSCComputeFeatures:
    @staticmethod
    def test_compute_features_gives_unchanged_features_when_centering_is_none():
        model = FewShotClassifier()
        features = torch.rand(10, 64)
        assert torch.allclose(model.compute_features(features), features)

    @staticmethod
    def test_compute_features_gives_centered_features_when_centering_is_not_none():
        model = FewShotClassifier(feature_centering=torch.rand(64))
        features = torch.rand(10, 64)
        assert torch.allclose(
            model.compute_features(features), features - model.feature_centering
        )
