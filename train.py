# train.py
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader # <-- New imports
from model import SimpleNet

from np_img_conversions import img_to_np

class AdaptiveEarlyStopper:
    """Early stopping that adjusts its patience dynamically and flags best models."""
    def __init__(self, base_patience, max_patience):
        self.base_patience = base_patience
        self.current_patience = base_patience
        self.max_patience = max_patience
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, current_loss):
        is_best = False
        significant_drop_threshold = self.best_loss * 0.995

        if current_loss < significant_drop_threshold:
            self.best_loss = current_loss
            self.counter = 0
            self.current_patience = max(self.base_patience, self.current_patience - 5)
            is_best = True
        elif current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter += 1
            self.current_patience = min(self.max_patience, int(self.current_patience * 1.1))
            is_best = True
        else:
            self.counter += 1

        stop_training = self.counter >= self.current_patience
        if stop_training:
            print(f"\n[Early Stop Triggered] Loss plateaued at {self.best_loss:.6f}.")
            print(f"Final adapted patience limit was {self.current_patience} epochs.")
            
        return stop_training, is_best


# --- NEW: PyTorch Dataset Class ---
class ImageDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        
        # Get sorted lists of files so input and target images align perfectly
        self.input_files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])
        self.target_files = sorted([f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))])
        
        # Sanity check
        assert len(self.input_files) == len(self.target_files), "Mismatch in number of input and target images!"

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        """Loads ONE image pair at a time, preventing RAM crashes."""
        in_path = os.path.join(self.input_dir, self.input_files[idx])
        tar_path = os.path.join(self.target_dir, self.target_files[idx])

        # Load, convert to float32, and normalize to 0.0 - 1.0
        in_np = img_to_np(in_path).astype(np.float32) / 255.0
        tar_np = img_to_np(tar_path).astype(np.float32) / 255.0

        # Convert (H, W, C) to (C, H, W)
        in_tensor = torch.tensor(in_np.transpose(2, 0, 1))
        tar_tensor = torch.tensor(tar_np.transpose(2, 0, 1))

        return in_tensor, tar_tensor


def main(config):
    # --- GPU SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize model, loss, and optimizer
    model = SimpleNet().to(device)
    criterion = nn.MSELoss()
    
    lr = config['training']['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 2. Setup Data Loading
    print("Initializing DataLoaders...")
    dataset = ImageDataset(config['data']['input_dir'], config['data']['target_dir'])
    
    # DataLoader handles batching and shuffling automatically
    batch_size = config['training']['batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Found {len(dataset)} image pairs. Batches per epoch: {len(dataloader)}")

    # 3. Training Loop Setup
    epochs = config['training']['epochs']
    print_freq = config['training']['print_freq']
    save_path = config['model']['save_path']
    
    early_stopper = AdaptiveEarlyStopper(
        base_patience=config['early_stopping']['base_patience'], 
        max_patience=config['early_stopping']['max_patience']
    )
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        
        # Track the total loss for the whole dataset this epoch
        epoch_loss = 0.0
        
        # -- INNER LOOP: Iterate over batches --
        for batch_in, batch_target in dataloader:
            # Move the specific batch to the GPU
            batch_in = batch_in.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            
            output = model(batch_in)
            loss = criterion(output, batch_target)
            
            loss.backward()
            optimizer.step()
            
            # Accumulate loss (multiply by batch size to get total loss for these items)
            epoch_loss += loss.item() * batch_in.size(0)
            
        # Calculate the average loss across all images this epoch
        avg_epoch_loss = epoch_loss / len(dataset)
        
        # Check early stopping using the AVERAGE loss of the epoch
        stop_training, is_best = early_stopper.step(avg_epoch_loss)
        
        # ONLY save the model if it achieved a new best loss
        if is_best:
            torch.save(model.state_dict(), save_path)
        
        if stop_training:
            print(f"Stopping at Epoch {epoch+1}. Best model weights are secured in '{save_path}'.")
            break
        
        if (epoch + 1) % print_freq == 0: 
            print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_epoch_loss:.6f} | Patience: {early_stopper.current_patience}")

    print("Training process finalized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimpleNet with a YAML config")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)
