# === baseline_classifier ===
# This script loads pre-computed spectrogram PNGs, extracts simple averaged-frame features,
# and trains a baseline PyTorch classifier (a small MLP) on them.

import os, glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SpectrogramDataset(Dataset):
    """
    PyTorch Dataset for loading spectrogram PNGs.
    Each sample is stored as <root_dir>/<digit>/<file>.png.
    We convert each image to grayscale, normalize, and then average
    over time frames (columns) to get a 128-dimensional feature vector.
    """
    def __init__(self, root_dir):
        self.samples = []
        # Walk through subdirectories '0' through '9'
        for digit in os.listdir(root_dir):
            path = os.path.join(root_dir, digit)
            if not os.path.isdir(path):
                continue
            # Collect all PNG files in this digit folder
            for img_path in glob.glob(os.path.join(path, '*.png')):
                # Store tuple (image_path, label)
                self.samples.append((img_path, int(digit)))

    def __len__(self):
        # Return total number of samples
        return len(self.samples)

    def __getitem__(self, idx):
        # Load a single sample by index
        img_path, label = self.samples[idx]
        # Open image and convert to grayscale ('L')
        img = Image.open(img_path).convert('L')
        # Convert to NumPy array and normalize to [0,1]
        arr = np.array(img, dtype=np.float32) / 255.0
        # Average across time (columns) to get one value per frequency bin
        feat = arr.mean(axis=1)  # shape: (128,)
        # Return a PyTorch tensor for features and integer label
        return torch.from_numpy(feat), label


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Takes a list of (feature_tensor, label) and stacks features into a tensor of shape
    (batch_size, 128), and labels into a 1D tensor.
    """
    feats, labels = zip(*batch)
    return torch.stack(feats), torch.tensor(labels)

class Net(nn.Module):
    """
    A simple two-layer MLP:
      128 inputs -> 64 hidden units -> 10 outputs (digits 0–9)
    Activation: ReLU
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)  # first linear layer
        self.relu = nn.ReLU()          # non-linearity
        self.fc2 = nn.Linear(64, 10)   # output layer for 10 classes

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

if __name__ == '__main__':
    # Initialize datasets for training and testing
    train_ds = SpectrogramDataset('Train_spectrograms')
    test_ds  = SpectrogramDataset('Test_spectrograms')

    # Print dataset sizes (number of samples)
    print(f"train_ds size: {len(train_ds)} samples")
    print(f"test_ds size:  {len(test_ds)} samples")

    # Inspect a single sample's feature-vector shape
    feat0, label0 = train_ds[0]
    print(f"Example train feature-vector shape: {feat0.shape}, label: {label0}")
    feat1, label1 = test_ds[0]
    print(f"Example test  feature-vector shape: {feat1.shape}, label: {label1}")

    # Inspect a single batch's shape from DataLoader
    temp_loader = DataLoader(train_ds, batch_size=32, collate_fn=collate_fn)
    batch_feats, batch_labels = next(iter(temp_loader))
    print(f"Batch feature-tensor shape: {batch_feats.shape}, labels: {batch_labels.shape}")

    # Create DataLoaders for training loop
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Instantiate model, loss function, and optimizer
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop: 20 epochs
    for epoch in range(1, 21):
        model.train()  # set model to training mode
        total_loss = 0.0
        for feats, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(feats)           # forward pass
            loss = criterion(outputs, labels)  # compute loss
            loss.backward()                    # backpropagate
            optimizer.step()                   # update weights
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/20 — Avg Loss: {avg_loss:.4f}")

    # Evaluation on test set
    model.eval()  # set model to evaluation mode
    correct = total = 0
    with torch.no_grad():  # no gradient computation
        for feats, labels in test_loader:
            outputs = model(feats)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100.0 * correct / total
    print(f"Baseline Classifier Accuracy: {accuracy:.2f}%")
