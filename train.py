# train.py
import os
import glob
import random
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
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


class ImageDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir

        self.input_files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])
        self.target_files = sorted([f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))])

        assert len(self.input_files) == len(self.target_files), "Mismatch in number of input and target images!"

        # Shuffle targets with a fixed seed so each input maps to a different orange.
        # Fixed seed ensures the pairing is stable across training runs.
        #rng = random.Random(42)
        #rng.shuffle(self.target_files)

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        """Loads ONE image pair and applies synchronized random augmentation."""
        in_path = os.path.join(self.input_dir, self.input_files[idx])
        tar_path = os.path.join(self.target_dir, self.target_files[idx])

        in_np = img_to_np(in_path).astype(np.float32) / 255.0
        tar_np = img_to_np(tar_path).astype(np.float32) / 255.0

        in_tensor = torch.tensor(in_np.transpose(2, 0, 1))
        tar_tensor = torch.tensor(tar_np.transpose(2, 0, 1))

        # Synchronized augmentation: same random flip applied to both tensors
        if random.random() < 0.5:
            in_tensor = TF.hflip(in_tensor)
            tar_tensor = TF.hflip(tar_tensor)
        if random.random() < 0.5:
            in_tensor = TF.vflip(in_tensor)
            tar_tensor = TF.vflip(tar_tensor)

        return in_tensor, tar_tensor


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleNet().to(device)
    criterion = nn.MSELoss()

    lr = config['training']['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = config['training']['epochs']
    min_lr = config['training']['min_lr']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    print("Initializing DataLoaders...")
    dataset = ImageDataset(config['data']['input_dir'], config['data']['target_dir'])
    batch_size = config['training']['batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Found {len(dataset)} image pairs. Batches per epoch: {len(dataloader)}")

    print_freq = config['training']['print_freq']
    save_path = config['model']['save_path']

    early_stopper = AdaptiveEarlyStopper(
        base_patience=config['early_stopping']['base_patience'],
        max_patience=config['early_stopping']['max_patience']
    )

    # Pick the first input image as the fixed probe for visualizing progress
    os.makedirs('frames', exist_ok=True)
    probe_file = sorted(os.listdir(config['data']['input_dir']))[0]
    probe_path = os.path.join(config['data']['input_dir'], probe_file)
    probe_np = img_to_np(probe_path).astype(np.float32) / 255.0
    probe_tensor = torch.tensor(probe_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    print(f"Probe image: {probe_file}")

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_in, batch_target in dataloader:
            batch_in = batch_in.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            output = model(batch_in)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_in.size(0)

        avg_epoch_loss = epoch_loss / len(dataset)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        stop_training, is_best = early_stopper.step(avg_epoch_loss)

        if is_best:
            torch.save(model.state_dict(), save_path)

        # Save a frame of the probe image prediction every epoch
        model.eval()
        with torch.no_grad():
            frame = model(probe_tensor).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        model.train()
        frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
        cv2.imwrite(f'frames/epoch_{epoch+1:04d}.png', cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))

        if stop_training:
            print(f"Stopping at Epoch {epoch+1}. Best model weights are secured in '{save_path}'.")
            break

        if (epoch + 1) % print_freq == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_epoch_loss:.6f} | LR: {current_lr:.7f} | Patience: {early_stopper.current_patience}")

    print("Training process finalized.")

    # Stitch saved frames into a video
    frame_files = sorted(glob.glob('frames/epoch_*.png'))
    if frame_files:
        sample = cv2.imread(frame_files[0])
        h, w = sample.shape[:2]
        writer = cv2.VideoWriter('training_progress.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        for f in frame_files:
            writer.write(cv2.imread(f))
        writer.release()
        print(f"Training progress video saved to training_progress.mp4 ({len(frame_files)} frames @ 30fps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimpleNet with a YAML config")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)
