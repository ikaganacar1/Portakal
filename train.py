# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import SimpleNet # Import the network from model.py

from np_img_conversions import img_to_np

class AdaptiveEarlyStopper:
    """
    Early stopping that adjusts its patience dynamically.
    """
    def __init__(self, base_patience=100, max_patience=400):
        self.base_patience = base_patience
        self.current_patience = base_patience
        self.max_patience = max_patience
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, current_loss):
        # Define a "significant" drop as a 0.5% relative improvement
        significant_drop_threshold = self.best_loss * 0.995

        if current_loss < significant_drop_threshold:
            # Huge improvement! Reset counter, tighten patience back to baseline.
            self.best_loss = current_loss
            self.counter = 0
            self.current_patience = max(self.base_patience, self.current_patience - 10)
            
        elif current_loss < self.best_loss:
            # Minor improvement. Update best loss, but don't fully reset the counter.
            # Multiply patience by 1.1 to give it more time to traverse this flat plateau.
            self.best_loss = current_loss
            self.counter += 1
            self.current_patience = min(self.max_patience, int(self.current_patience * 1.1))
            
        else:
            # No improvement at all
            self.counter += 1

        # Check if we've run out of patience
        if self.counter >= self.current_patience:
            print(f"\n[Early Stop Triggered] Loss plateaued at {self.best_loss:.4f}.")
            print(f"Final adapted patience limit was {self.current_patience} epochs.")
            return True
            
        return False

def load_data(directory: str):
    """Reads all images from a directory and returns a list of NumPy arrays."""
    data = []
    for img in sorted(os.listdir(directory)):
        full_path = os.path.join(directory, img)
        if os.path.isfile(full_path):
            img_array = img_to_np(full_path)
            data.append(img_array.astype(np.float32))
            
    return data

def main():
    # 1. Initialize model, loss, and optimizer
    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 2. Load the data
    inputs_list = load_data('orange_oranges')
    targets_list = load_data('orange_oranges')

    # Extract first element and normalize to 0.0 - 1.0 range!
    input_np = inputs_list[0] / 255.0
    target_np = targets_list[0] / 255.0

    # Convert NumPy (H, W, C) to PyTorch (Batch, C, H, W)
    input_tensor = torch.tensor(input_np.transpose(2, 0, 1)).unsqueeze(0)
    target_tensor = torch.tensor(target_np.transpose(2, 0, 1)).unsqueeze(0)

    # 3. Training Loop
    epochs = 5000
    early_stopper = AdaptiveEarlyStopper(base_patience=100, max_patience=400)
    
    print("Starting training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output_tensor = model(input_tensor)
        loss = criterion(output_tensor, target_tensor)
        loss.backward()
        optimizer.step()
        
        # Check early stopping at the end of every epoch
        if early_stopper.step(loss.item()):
            print(f"Stopping at Epoch {epoch+1} to prevent overfitting/wasting compute.")
            break
        
        if (epoch + 1) % 50 == 0: # Adjusted print frequency since we have 5000 epochs
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f} | Current Patience: {early_stopper.current_patience}")

    # 4. Save the trained model weights
    save_path = 'simple_net.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model weights saved to '{save_path}'")

if __name__ == "__main__":
    main()
