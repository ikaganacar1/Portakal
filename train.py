# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import SimpleNet # Import the network from model.py

def main():
    # 1. Initialize model, loss, and optimizer
    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 2. Create Dummy Training Data
    print("Generating dummy training data...")
    input_np = np.random.rand(256, 256, 3).astype(np.float32)
    target_np = np.random.rand(256, 256, 3).astype(np.float32)

    # Convert NumPy (H, W, C) to PyTorch (Batch, C, H, W)
    input_tensor = torch.tensor(input_np.transpose(2, 0, 1)).unsqueeze(0)
    target_tensor = torch.tensor(target_np.transpose(2, 0, 1)).unsqueeze(0)

    # 3. Training Loop
    epochs = 50
    print("Starting training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output_tensor = model(input_tensor)
        loss = criterion(output_tensor, target_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 4. Save the trained model weights
    save_path = 'simple_net.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model weights saved to '{save_path}'")

if __name__ == "__main__":
    main()
