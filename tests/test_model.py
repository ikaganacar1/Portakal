# tests/test_model.py
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import SimpleNet

def test_output_shape():
    model = SimpleNet()
    x = torch.rand(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 3, 256, 256), f"Expected (2, 3, 256, 256), got {out.shape}"

def test_output_range():
    model = SimpleNet()
    x = torch.rand(2, 3, 256, 256)
    out = model(x)
    assert out.min().item() >= 0.0, f"Output min {out.min().item()} is below 0"
    assert out.max().item() <= 1.0, f"Output max {out.max().item()} is above 1"

def test_output_not_saturated():
    """Ensure output is not all-white (all 1.0) or all-black (all 0.0)."""
    model = SimpleNet()
    x = torch.rand(2, 3, 256, 256)
    out = model(x)
    assert out.std().item() > 0.01, f"Output appears saturated (std={out.std().item():.4f})"
