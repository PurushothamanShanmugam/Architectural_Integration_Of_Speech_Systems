import torch
from src.lid.model import MultiHeadLID

def test_lid_model_output_shape():
    model = MultiHeadLID()
    x = torch.randn(2, 100, 40)
    y = model(x)
    assert tuple(y.shape) == (2, 100, 2)
