# model.py
import torch.nn as nn

class SimpleNet(nn.Module):
    """
    A deeper convolutional neural network designed to map features 
    while preserving the (3, 256, 256) shape.
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        
        self.net = nn.Sequential(
            # Extract features
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(), # Adds non-linearity
            
            # Process features
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Map back to 3 RGB channels
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1),
            #nn.Sigmoid() # Forces all output values to be strictly between 0.0 and 1.0
        )

    def forward(self, x):
        return self.net(x)
