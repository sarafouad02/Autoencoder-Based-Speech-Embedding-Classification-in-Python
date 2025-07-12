# === script: ae_sliding_window_embeddings.py ===

import os, glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── 1) Helpers ───────────────────────────────────────────────────────────────
FRAME_DIM = 128

def extract_frames(img_path):
    """
    Load a spectrogram image, normalize to [0,1],
    and return an array (T, 128) where each row is one time frame.
    """
    img = Image.open(img_path).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0  # (128, T)
    return arr.T  # (T, 128)

# ── 2) AE Dataset: concat every pair of consecutive frames ───────────────────
class PairFrameDataset(Dataset):
    def __init__(self, root_dir):
        self.pairs = []  # list of (x, x) for AE training
        for digit in sorted(os.listdir(root_dir)):
            dpath = os.path.join(root_dir, digit)
            if not os.path.isdir(dpath): continue
            for png in glob.glob(os.path.join(dpath, '*.png')):
                frames = extract_frames(png)            # (T,128)
                for i in range(frames.shape[0] - 1):
                    f1, f2 = frames[i], frames[i+1]     # two consecutive frames
                    x = np.concatenate([f1, f2])        # (256,)
                    self.pairs.append(x)
        self.pairs = np.stack(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.pairs[idx]).float()
        # AE target is reconstructing the same concatenated input
        return x, x

# ── 3) Autoencoder: input=256 → latent=128 → recon=256 ─────────────────────
class AE(nn.Module):
    def __init__(self, frame_dim=FRAME_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(frame_dim*2, 256),
            nn.ReLU(),
            nn.Linear(256, frame_dim),    # latent size = 128
        )
        self.decoder = nn.Sequential(
            nn.Linear(frame_dim, 256),
            nn.ReLU(),
            nn.Linear(256, frame_dim*2),
            nn.Sigmoid(),                 # inputs were in [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

# ── 4) Train AE ──────────────────────────────────────────────────────────────
def train_ae(ae, loader, epochs=20, lr=1e-3, device='cpu'):
    ae.to(device)
    opt = optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for ep in range(1, epochs+1):
        ae.train()
        total = 0
        for x, x_target in loader:
            x, x_target = x.to(device), x_target.to(device)
            x_rec, _ = ae(x)
            loss = criterion(x_rec, x_target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"[AE] Epoch {ep}/{epochs} — MSE: {total/len(loader):.6f}")

# ── 5) Build utterance embedding via sliding-window through encoder ─────────
def utterance_embedding(encoder, img_path, device='cpu'):
    """
    1) Load frames (T,128).
    2) Take frames[0] & frames[1], concat → encode → z1 (128-D).
    3) Then for each i=2…T-1: concat(prev_z, frames[i]) → encode → z2, etc.
    4) Return the mean of all z‘s as the 128-D utterance vector.
    """
    frames = extract_frames(img_path)
    T = frames.shape[0]
    if T < 2:
        # too short: just average real frames
        return frames.mean(axis=0)

    encoder.to(device).eval()
    zs = []
    # first step: real frames 0 & 1
    prev1 = frames[0]
    prev2 = frames[1]
    with torch.no_grad():
        # initial latent
        inp = torch.from_numpy(np.concatenate([prev1, prev2])).float().unsqueeze(0).to(device)
        z = encoder(inp).squeeze(0).cpu().numpy()
        zs.append(z)
        # roll forward
        for i in range(2, T):
            inp = torch.from_numpy(np.concatenate([z, frames[i]])).float().unsqueeze(0).to(device)
            z = encoder(inp).squeeze(0).cpu().numpy()
            zs.append(z)

    return np.stack(zs, axis=0).mean(axis=0)  # (128,)

# ── 6) Classifier on 128-D utterance vectors ────────────────────────────────
class Classifier(nn.Module):
    def __init__(self, input_dim=FRAME_DIM, hidden=64, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_classifier(clf, X_tr, y_tr, X_te, y_te, epochs=20, lr=1e-3, device='cpu'):
    clf.to(device)
    opt = optim.Adam(clf.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr).float(),
                                        torch.from_numpy(y_tr).long())
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    for ep in range(1, epochs+1):
        clf.train()
        tot = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = clf(xb)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"[CLF] Epoch {ep}/{epochs} — CE: {tot/len(loader):.4f}")

    # final accuracy
    clf.eval()
    with torch.no_grad():
        logits = clf(torch.from_numpy(X_te).float().to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    acc = (preds == y_te).mean() * 100
    print(f"[CLF] Test Acc: {acc:.2f}%")

# ── 7) Main: tie it all together ────────────────────────────────────────────
if __name__ == '__main__':
    TRAIN_DIR = 'Train_spectrograms'
    TEST_DIR  = 'Test_spectrograms'
    DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) AE training data
    print("Preparing AE dataset…")
    ae_ds = PairFrameDataset(TRAIN_DIR)
    ae_loader = DataLoader(ae_ds, batch_size=32, shuffle=True)

    # 2) Train AE
    print("Training AE on frame‐pairs…")
    ae = AE(frame_dim=FRAME_DIM)
    train_ae(ae, ae_loader, epochs=20, lr=1e-3, device=DEVICE)

    # 3) Build utterance embeddings
    print("Computing utterance embeddings…")
    # freeze encoder
    encoder = ae.encoder

    X_train, y_train = [], []
    for digit in sorted(os.listdir(TRAIN_DIR)):
        for png in glob.glob(os.path.join(TRAIN_DIR, digit, '*.png')):
            emb = utterance_embedding(encoder, png, device=DEVICE)
            X_train.append(emb); y_train.append(int(digit))
    X_test, y_test = [], []
    for digit in sorted(os.listdir(TEST_DIR)):
        for png in glob.glob(os.path.join(TEST_DIR, digit, '*.png')):
            emb = utterance_embedding(encoder, png, device=DEVICE)
            X_test.append(emb); y_test.append(int(digit))

    X_train = np.stack(X_train)
    y_train = np.array(y_train)
    X_test  = np.stack(X_test)
    y_test  = np.array(y_test)
    print(f"Train embeddings: {X_train.shape}, Test embeddings: {X_test.shape}")

    # 4) Train + evaluate classifier
    print("Training classifier on 128‑D utterance vectors…")
    clf = Classifier(input_dim=FRAME_DIM, hidden=64, n_classes=10)
    train_classifier(clf, X_train, y_train, X_test, y_test,
                     epochs=20, lr=1e-3, device=DEVICE)
