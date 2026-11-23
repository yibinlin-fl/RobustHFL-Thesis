import pytest
import torch

from core.models import build_model


@pytest.mark.parametrize(("model_name", "dimension"), [
    ("ResNet10", 512), ("ResNet12", 512), ("ShuffleNet", 800), ("MobileNetV2", 1280),
])
def test_cifar_forward(model_name, dimension):
    torch.set_num_threads(2)
    x = torch.rand(4, 3, 32, 32)
    model = build_model(model_name).eval()
    with torch.inference_mode():
        logits, features = model(x, return_feat=True)
        assert logits.shape == (4, 10)
        assert features.shape == (4, dimension)
        torch.testing.assert_close(features.norm(dim=1), torch.ones(4))
        torch.testing.assert_close(logits, model(x))
