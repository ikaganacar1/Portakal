# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import SimpleNet # Import the network from model.py

from np_img_conversions import img_to_np

def load_data(directory: str):
    """Reads all images from a directory and returns a list of NumPy arrays."""
    data = []
    # Sorting ensures that if you have matching filenames in input/target dirs, they align
    for img in sorted(os.listdir(directory)):
        full_path = os.path.join(directory, img)
        
        # Check if it's a file to avoid crashing on subdirectories
        if os.path.isfile(full_path):
            img_array = img_to_np(full_path)
            # Ensure the array is float32, which PyTorch generally expects for model weights
            data.append(img_array.astype(np.float32))
            
    return data

def main():
    # 1. Initialize model, loss, and optimizer
    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 2. Load the data
    # (Update these directory paths to match your local file structure)
    inputs_list = load_data('orange_oranges')
    targets_list = load_data('orange_oranges')

    # To maintain your current setup (training on a single image), 
    # we'll extract the first element from the loaded lists.
    input_np = inputs_list[0]
    target_np = targets_list[0]

    # Convert NumPy (H, W, C) to PyTorch (Batch, C, H, W)
    input_tensor = torch.tensor(input_np.transpose(2, 0, 1)).unsqueeze(0)
    target_tensor = torch.tensor(target_np.transpose(2, 0, 1)).unsqueeze(0)

    # 3. Training Loop
    epochs = 5000
    print("Starting training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output_tensor = model(input_tensor)
        loss = criterion(output_tensor, target_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 4. Save the trained model weights
    save_path = 'simple_net.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model weights saved to '{save_path}'")

if __name__ == "__main__":
    main()
