# === script: ae_min_error_embeddings.py ===

import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── 1) Constants & helper to load per-frame vectors ───────────────────────
FRAME_DIM = 128

def extract_frames(img_path):
    """
    Load a spectrogram PNG, normalize to [0,1],
    return array shape (T,128), one row per time-frame.
    """
    img = Image.open(img_path).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0  # (128, T)
    return arr.T  # (T, 128)

# ── 2) Dataset: every consecutive‐frame pair x → x for AE training ───────
class PairFrameDataset(Dataset):
    def __init__(self, root_dir):
        self.pairs = []
        for digit in sorted(os.listdir(root_dir)):
            dpath = os.path.join(root_dir, digit)
            if not os.path.isdir(dpath): continue
            for png in glob.glob(os.path.join(dpath, '*.png')):
                frames = extract_frames(png)  # (T,128)
                for i in range(frames.shape[0] - 1):
                    f1, f2 = frames[i], frames[i+1]
                    self.pairs.append(np.concatenate([f1, f2]))  # (256,)
        self.pairs = np.stack(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.pairs[idx]).float()
        return x, x   # AE target = reconstruct same input

# ── 3) Autoencoder: 256→128→256 ───────────────────────────────────────────
class AE(nn.Module):
    def __init__(self, frame_dim=FRAME_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(frame_dim*2, 256),
            nn.ReLU(),
            nn.Linear(256, frame_dim),     # latent size = 128
        )
        self.decoder = nn.Sequential(
            nn.Linear(frame_dim, 256),
            nn.ReLU(),
            nn.Linear(256, frame_dim*2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

# ── 4) AE training loop ────────────────────────────────────────────────────
def train_ae(ae, loader, epochs=20, lr=1e-3, device='cpu'):
    ae.to(device)
    opt = optim.Adam(ae.parameters(), lr=lr)
    crit = nn.MSELoss()
    for ep in range(1, epochs+1):
        ae.train()
        tot = 0.0
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            rec, _ = ae(x)
            loss = crit(rec, target)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"[AE] Epoch {ep}/{epochs} — MSE {tot/len(loader):.6f}")

# ── 5) Utterance embedding with min-error sliding window ─────────────────
def utterance_embedding_min_error(ae, img_path, device='cpu'):
    """
    At each frame i, test 4 candidates:
      A = [prev_z,       f_i]
      B = [prev_frame,   f_i]
      C = [prev_z,       prev_prev_z]
      D = [f_i,          prev_frame]

    Print which candidate has the lowest AE reconstruction error,
    then encode it to z_i and proceed.
    """
    frames = extract_frames(img_path)       # (T,128)
    T = frames.shape[0]
    if T < 2:
        return frames.mean(axis=0)

    ae.to(device).eval()
    with torch.no_grad():
        # ── Initial step: real frames 0 & 1
        f0, f1 = frames[0], frames[1]
        inp0 = torch.from_numpy(np.concatenate([f0, f1]))\
                  .float().unsqueeze(0).to(device)
        _, z1 = ae(inp0)
        prev_prev_z = prev_z = z1.squeeze(0).cpu().numpy()
        prev_frame  = f1.copy()
        zs = [prev_z]

        # ── Rolling through the rest of the frames
        for i in range(2, T):
            fi = frames[i]
            # build four candidates
            cands = {
                'A': np.concatenate([prev_z,     fi]),           # latent + current
                'B': np.concatenate([prev_frame, fi]),           # real prev + current
                'C': np.concatenate([prev_z,     prev_prev_z]),  # prev latent + prev-prev latent
                # 'D': np.concatenate([fi,         prev_frame]),   # current + real prev
            }

            best_err = float('inf')
            best_z   = None
            best_key = None

            # evaluate all
            for key, x_np in cands.items():
                x_t = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
                rec, z_t = ae(x_t)
                err = ((rec.cpu().numpy().squeeze() - x_np)**2).mean()
                if err < best_err:
                    best_err = err
                    best_z   = z_t.squeeze(0).cpu().numpy()
                    best_key = key

            # print which candidate won
            print(f"Frame {i}: best option = {best_key}, error = {best_err:.6f}")

            # roll forward
            prev_prev_z = prev_z
            prev_z      = best_z
            prev_frame  = fi
            zs.append(prev_z)

    return np.stack(zs, axis=0).mean(axis=0)



# ── 6) Classifier & training ──────────────────────────────────────────────
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

def train_classifier(clf, X_tr, y_tr, X_te, y_te,
                     epochs=20, lr=1e-3, device='cpu'):
    clf.to(device)
    opt = optim.Adam(clf.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(y_tr).long()
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    for ep in range(1, epochs+1):
        clf.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = clf(xb)
            loss   = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"[CLF] Epoch {ep}/{epochs} — CE {tot/len(loader):.4f}")

    clf.eval()
    with torch.no_grad():
        logits = clf(torch.from_numpy(X_te).float().to(device))
        preds  = logits.argmax(dim=1).cpu().numpy()
    acc = (preds == y_te).mean() * 100
    print(f"[CLF] Test Accuracy: {acc:.2f}%")

# ── 7) Main: train AE → build min-error embeddings → classify ────────────
if __name__ == '__main__':
    TRAIN_DIR = 'Train_spectrograms'
    TEST_DIR  = 'Test_spectrograms'
    DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

    # AE training
    print("Preparing AE dataset…")
    ae_ds = PairFrameDataset(TRAIN_DIR)
    ae_loader = DataLoader(ae_ds, batch_size=32, shuffle=True)

    print("Training AE on frame-pairs…")
    ae = AE(frame_dim=FRAME_DIM)
    train_ae(ae, ae_loader, epochs=20, lr=1e-3, device=DEVICE)

    # Build utterance embeddings with minimum-error sliding
    print("Computing min-error utterance embeddings…")
    X_train, y_train = [], []
    for digit in sorted(os.listdir(TRAIN_DIR)):
        for png in glob.glob(os.path.join(TRAIN_DIR, digit, '*.png')):
            emb = utterance_embedding_min_error(ae, png, device=DEVICE)
            X_train.append(emb); y_train.append(int(digit))

    X_test, y_test = [], []
    for digit in sorted(os.listdir(TEST_DIR)):
        for png in glob.glob(os.path.join(TEST_DIR, digit, '*.png')):
            emb = utterance_embedding_min_error(ae, png, device=DEVICE)
            X_test.append(emb); y_test.append(int(digit))

    X_train = np.stack(X_train); y_train = np.array(y_train)
    X_test  = np.stack(X_test);  y_test  = np.array(y_test)
    print(f"Train embeddings: {X_train.shape}, Test embeddings: {X_test.shape}")

    # Classifier training & evaluation
    print("Training classifier on min-error embeddings…")
    clf = Classifier(input_dim=FRAME_DIM, hidden=64, n_classes=10)
    train_classifier(clf, X_train, y_train, X_test, y_test,
                     epochs=20, lr=1e-3, device=DEVICE)
