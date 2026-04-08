# model.py
import torch.nn as nn

class SimpleNet(nn.Module):
    """
    Fully convolutional network: maps (3, 256, 256) -> (3, 256, 256).
    64 channels + BatchNorm for stability. Sigmoid constrains output to [0, 1].
    """
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.net = nn.Sequential(
            # Extract features
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Process features
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Map back to 3 RGB channels
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Constrains output to [0, 1] matching normalized targets
        )

    def forward(self, x):
        return self.net(x)
