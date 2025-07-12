# === autoencoder_concat ===

import os, glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------------
# 1) Dataset that pads each spectrogram to the max number of frames
# -------------------------------------------------------------------
class ConcatSpectrogramDataset(Dataset):
    def __init__(self, root_dir, max_frames=None):
        self.samples = []
        self.max_frames = 0
        for digit in os.listdir(root_dir):
            dpath = os.path.join(root_dir, digit)
            if not os.path.isdir(dpath): continue
            for img in glob.glob(os.path.join(dpath, '*.png')):
                self.samples.append((img, int(digit)))

        # compute max_frames if not provided
        if max_frames is None:
            for img_path, _ in self.samples:
                img = Image.open(img_path).convert('L')
                arr = np.array(img, dtype=np.float32) / 255.0
                self.max_frames = max(self.max_frames, arr.shape[1])
        else:
            self.max_frames = max_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
        # pad with zeros on the right (time dimension) up to max_frames
        pad_width = self.max_frames - arr.shape[1]
        if pad_width > 0:
            arr = np.pad(arr, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
        # flatten to a single vector of length 128 * max_frames
        feat = arr.flatten()
        return torch.from_numpy(feat), label

# -------------------------------------------------------------------
# 2) Simple Autoencoder
# -------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=256):
        super().__init__()
        # encoder: input_dim -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        # decoder: latent_dim -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),  # since inputs are in [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

# -------------------------------------------------------------------
# 3) Simple classifier on top of the frozen encoder
# -------------------------------------------------------------------
# class Classifier(nn.Module):
#     def __init__(self, latent_dim=256, n_classes=10):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(latent_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, n_classes),
#         )
#     def forward(self, z):
#         return self.net(z)
    
class Classifier(nn.Module):
    """
    A simple two-layer MLP:
      128 inputs -> 64 hidden units -> 10 outputs (digits 0–9)
    Activation: ReLU
    """
    def __init__(self, latent_dim=256, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# -------------------------------------------------------------------
# 4) Training / evaluation helpers
# -------------------------------------------------------------------
def train_autoencoder(ae, loader, epochs=20, lr=1e-3, device='cpu'):
    ae.to(device)
    opt = optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for ep in range(1, epochs+1):
        ae.train()
        running_loss = 0
        for x, _ in loader:
            x = x.to(device)
            x_rec, _ = ae(x)
            loss = criterion(x_rec, x)
            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += loss.item()
        print(f"[AE] Epoch {ep}/{epochs}, MSE Loss: {running_loss/len(loader):.6f}")

def extract_latents(ae, loader, device='cpu'):
    ae.to(device).eval()
    latents, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, z = ae(x)
            latents.append(z.cpu())
            labels.append(y)
    return torch.cat(latents, dim=0), torch.cat(labels, dim=0)

def train_classifier(clf, z_train, y_train, z_test, y_test, epochs=20, lr=1e-3, device='cpu'):
    clf.to(device)
    opt = optim.Adam(clf.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # wrap in TensorDatasets if desired; we'll do simple batches
    dataset = torch.utils.data.TensorDataset(z_train, y_train)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)
    for ep in range(1, epochs+1):
        clf.train()
        total_loss = 0
        for z_batch, y_batch in loader:
            z_batch, y_batch = z_batch.to(device), y_batch.to(device)
            logits = clf(z_batch)
            loss = criterion(logits, y_batch)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"[CLF] Epoch {ep}/{epochs}, CE Loss: {total_loss/len(loader):.4f}")

    # final accuracy
    clf.eval()
    with torch.no_grad():
        logits = clf(z_test.to(device))
        preds = logits.argmax(dim=1).cpu()
        acc = (preds == y_test).float().mean().item()
    print(f"[CLF] Test Accuracy: {100*acc:.2f}%")

# -------------------------------------------------------------------
# 5) Main: load data, train AE, extract latents, train classifier
# -------------------------------------------------------------------
if __name__ == '__main__':
    # create datasets + loaders
    train_ds_tmp = ConcatSpectrogramDataset('Train_spectrograms')
    test_ds_tmp  = ConcatSpectrogramDataset('Test_spectrograms')

    common_max_frames = max(train_ds_tmp.max_frames, test_ds_tmp.max_frames)

    # now create final datasets with consistent padding
    train_ds = ConcatSpectrogramDataset('Train_spectrograms', max_frames=common_max_frames)
    test_ds  = ConcatSpectrogramDataset('Test_spectrograms', max_frames=common_max_frames)

    print(f"Max frames per utterance: {train_ds.max_frames}")
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # DataLoaders for AE (we only need x, not labels)
    train_ae_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_ae_loader  = DataLoader(test_ds,  batch_size=32)

    # instantiate and train autoencoder
    input_dim = 128 * train_ds.max_frames
    ae = Autoencoder(input_dim=input_dim, latent_dim=256)
    train_autoencoder(ae, train_ae_loader, epochs=20, lr=1e-3)

    # extract latent codes for train + test
    z_train, y_train = extract_latents(ae, train_ae_loader)
    z_test,  y_test  = extract_latents(ae, test_ae_loader)

    print(f"Latent shapes: z_train={z_train.shape}, z_test={z_test.shape}")

    # train & evaluate a simple classifier on the latents
    clf = Classifier(latent_dim=256, n_classes=10)
    train_classifier(clf, z_train, y_train, z_test, y_test, epochs=20, lr=1e-3)
