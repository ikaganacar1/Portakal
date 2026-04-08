# model.py
import torch.nn as nn

class SimpleNet(nn.Module):
    """
    A simple convolutional neural network designed to take a 
    (3, 256, 256) PyTorch tensor and output the exact same shape.
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        # A single convolutional layer that preserves spatial dimensions and channels
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)
