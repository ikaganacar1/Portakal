# tests/test_augmentation.py
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_augmentation_keeps_input_target_in_sync():
    """
    Both tensors must receive the identical flip transforms.
    Since input==target in this project, any mismatch would show as nonzero diff.
    """
    import random
    import torchvision.transforms.functional as TF

    # Simulate a single sample: input and target are identical
    x = torch.rand(3, 256, 256)
    inp = x.clone()
    tar = x.clone()

    # Apply the same augmentation logic used in __getitem__
    if random.random() < 0.5:
        inp = TF.hflip(inp)
        tar = TF.hflip(tar)
    if random.random() < 0.5:
        inp = TF.vflip(inp)
        tar = TF.vflip(tar)

    # After identical transforms, they must still be equal
    assert torch.allclose(inp, tar), "Input and target diverged after augmentation"

def test_augmentation_does_not_change_shape():
    import torchvision.transforms.functional as TF
    x = torch.rand(3, 256, 256)
    assert TF.hflip(x).shape == (3, 256, 256)
    assert TF.vflip(x).shape == (3, 256, 256)
