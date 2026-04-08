# Orange Model Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix training instability and improve model capacity so the network reliably learns identity mapping on orange images without diverging to white output.

**Architecture:** Three focused changes — (1) update config hyperparameters, (2) upgrade model with more channels + BatchNorm + Sigmoid, (3) add LR scheduler and synchronized augmentation to training loop. Each change is independent and tested before the next.

**Tech Stack:** PyTorch, torchvision, PyYAML, pytest, numpy

---

### Task 1: Update config.yaml hyperparameters

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Update config.yaml**

Replace the full contents of `config.yaml` with:

```yaml
# config.yaml

data:
  input_dir: "colored_oranges"
  target_dir: "target_oranges"

training:
  epochs: 1000
  batch_size: 4
  learning_rate: 0.0001
  min_lr: 0.000001
  scheduler: cosine
  print_freq: 10

early_stopping:
  base_patience: 30
  max_patience: 100

model:
  save_path: "simple_net.pth"
```

- [ ] **Step 2: Verify config loads cleanly**

Run:
```bash
python -c "import yaml; c = yaml.safe_load(open('config.yaml')); print(c['training'])"
```
Expected output:
```
{'epochs': 1000, 'batch_size': 4, 'learning_rate': 0.0001, 'min_lr': 1e-06, 'scheduler': 'cosine', 'print_freq': 10}
```

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "config: lower lr to 0.0001, add cosine scheduler, increase epochs to 1000"
```

---

### Task 2: Upgrade model architecture

**Files:**
- Modify: `model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Create test file**

Create `tests/test_model.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail (model not yet updated)**

```bash
cd /mnt/2tb_ssd/Portakal && python -m pytest tests/test_model.py -v
```
Expected: `test_output_range` FAILS because Sigmoid is disabled and outputs can exceed 1.0.

- [ ] **Step 3: Update model.py**

Replace the full contents of `model.py` with:

```python
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
            nn.BatchNorm2d(3),
            nn.Sigmoid(),  # Constrains output to [0, 1] matching normalized targets
        )

    def forward(self, x):
        return self.net(x)
```

- [ ] **Step 4: Run tests — all must pass**

```bash
cd /mnt/2tb_ssd/Portakal && python -m pytest tests/test_model.py -v
```
Expected:
```
tests/test_model.py::test_output_shape PASSED
tests/test_model.py::test_output_range PASSED
tests/test_model.py::test_output_not_saturated PASSED
```

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "model: channels 16->64, add BatchNorm2d, re-enable Sigmoid"
```

---

### Task 3: Add LR scheduler and synchronized augmentation

**Files:**
- Modify: `train.py`
- Create: `tests/test_augmentation.py`

- [ ] **Step 1: Create augmentation test**

Create `tests/test_augmentation.py`:

```python
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
```

- [ ] **Step 2: Run augmentation tests — they should pass already (pure logic test)**

```bash
cd /mnt/2tb_ssd/Portakal && python -m pytest tests/test_augmentation.py -v
```
Expected: Both tests PASS (this validates the augmentation approach before wiring it in).

- [ ] **Step 3: Update train.py**

Replace the full contents of `train.py` with:

```python
# train.py
import os
import random
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
        scheduler.step()

        stop_training, is_best = early_stopper.step(avg_epoch_loss)

        if is_best:
            torch.save(model.state_dict(), save_path)

        if stop_training:
            print(f"Stopping at Epoch {epoch+1}. Best model weights are secured in '{save_path}'.")
            break

        if (epoch + 1) % print_freq == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_epoch_loss:.6f} | LR: {current_lr:.7f} | Patience: {early_stopper.current_patience}")

    print("Training process finalized.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimpleNet with a YAML config")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(config)
```

- [ ] **Step 4: Run all tests to confirm nothing broke**

```bash
cd /mnt/2tb_ssd/Portakal && python -m pytest tests/ -v
```
Expected:
```
tests/test_augmentation.py::test_augmentation_keeps_input_target_in_sync PASSED
tests/test_augmentation.py::test_augmentation_does_not_change_shape PASSED
tests/test_model.py::test_output_shape PASSED
tests/test_model.py::test_output_range PASSED
tests/test_model.py::test_output_not_saturated PASSED
```

- [ ] **Step 5: Smoke-test the training loop (5 epochs)**

```bash
cd /mnt/2tb_ssd/Portakal && python train.py --config config.yaml
```
Watch the first few epochs. Loss should decrease (not spike to NaN or stay flat). You should see output like:
```
Using device: cuda
Found 256 image pairs. Batches per epoch: 64
Starting training for 1000 epochs...
Epoch [10/1000], Avg Loss: 0.0XXXXX | LR: 0.0000999 | Patience: 30
```
If loss is NaN, stop — something is wrong. If loss is decreasing, training is healthy.

- [ ] **Step 6: Commit**

```bash
git add train.py tests/test_augmentation.py
git commit -m "train: add CosineAnnealingLR scheduler and synchronized hflip/vflip augmentation"
```

---

### Task 4: Run full training and verify output

**Files:**
- No code changes — this is a verification task

- [ ] **Step 1: Run full training**

```bash
cd /mnt/2tb_ssd/Portakal && python train.py --config config.yaml
```
Let it run to completion or early stopping. Expected final loss: below 0.005 for identity mapping.

- [ ] **Step 2: Run inference**

```bash
cd /mnt/2tb_ssd/Portakal && python infer.py
```

- [ ] **Step 3: Inspect result.png**

Open `result.png`. It should show a recognizable orange — not white, not black, not uniform color. The image may be slightly blurry, but it should be clearly orange-shaped with visible structure.

- [ ] **Step 4: Commit final model weights (optional)**

```bash
git add simple_net.pth
git commit -m "chore: save trained model weights after successful run"
```
