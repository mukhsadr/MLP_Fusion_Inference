#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, log_loss

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Settings
# -------------------------------------------------------------
BASE = "/home-local/sadridm/Projects/Spleen/ABNL-MARRO"
PIPELINES = ["TotalSegmentator", "DeepSpleenSeg", "DeepMultiOrgSeg", "GennUNet"]
PATCH_SIZE, MAX_VOXELS_PER_CASE = 9, 3000
USE_COORD_FEATURES = True
BALANCE_DATASET = True

N_EPOCHS = 80
BATCH_SIZE = 2048
LR = 5e-5
WEIGHT_DECAY = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", DEVICE)

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
def extract_patch(vol, center, size):
    r = size // 2
    x, y, z = center
    if any(i - r < 0 or i + r >= s for i, s in zip(center, vol.shape)):
        return None
    return vol[x-r:x+r+1, y-r:y+r+1, z-r:z+r+1].flatten()


def load_case(case):
    case_dir = os.path.join(BASE, case)
    ct = nib.load(os.path.join(case_dir, "INPUTS", "image.nii.gz")).get_fdata()
    label = nib.load(os.path.join(case_dir, "OUTPUTS", "label.nii.gz")).get_fdata()

    masks = []
    for name in PIPELINES:
        p = os.path.join(case_dir, "OUTPUTS", name, "spleen.nii.gz")
        if os.path.exists(p):
            m = nib.load(p).get_fdata()
            masks.append((m == 1).astype(np.uint8))
        else:
            masks.append(np.zeros_like(label))
    return ct, label, masks

# -------------------------------------------------------------
# ⭐ FIXED: Real Background Sampling
# -------------------------------------------------------------
def build_dataset(cases, patch_size=9, max_voxels=2000, balance=False):
    X, y = [], []

    for case in tqdm(cases, desc="Building dataset"):
        try:
            ct, label, masks = load_case(case)
        except Exception as e:
            print(f"⚠️ Skipping {case}: {e}")
            continue

        union = np.clip(sum(masks), 0, 1)

        # ------------------------------
        # Foreground (positive samples)
        # ------------------------------
        fg = np.argwhere(label == 1)

        if len(fg) > max_voxels:
            fg = fg[np.random.choice(len(fg), max_voxels, replace=False)]

        # ------------------------------
        # Background (HARD negatives)
        # region where pipelines disagree
        # ------------------------------
        disagreement = np.argwhere(
            (sum(masks) > 0) & (label == 0)
        )

        # fallback: random background
        if len(disagreement) < max_voxels:
            bg = np.argwhere(label == 0)
        else:
            bg = disagreement

        if len(bg) > max_voxels:
            bg = bg[np.random.choice(len(bg), max_voxels, replace=False)]

        coords = np.concatenate([fg, bg])

        # ------------------------------
        # Extract features
        # ------------------------------
        for idx in coords:
            patch = extract_patch(ct, tuple(idx), patch_size)
            if patch is None:
                continue

            patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-8)
            feats = [patch.astype(np.float32)]

            for m in masks:
                mp = extract_patch(m, tuple(idx), patch_size)
                feats.append(mp.astype(np.float32))

            feats.append((np.array(idx) / np.array(ct.shape)).astype(np.float32))

            X.append(np.concatenate(feats))
            y.append(int(label[tuple(idx)] == 1))

    X, y = np.array(X), np.array(y)

    # Balanced sampling
    if balance:
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        n = min(len(pos), len(neg))
        idx = np.concatenate([
            np.random.choice(pos, n, replace=False),
            np.random.choice(neg, n, replace=False)
        ])
        X, y = X[idx], y[idx]

    return X, y

# -------------------------------------------------------------
# Load datasets
# -------------------------------------------------------------
all_cases = sorted([d for d in os.listdir(BASE) if d.startswith("001-")])
train_cases, temp_cases = train_test_split(all_cases, test_size=0.3, random_state=42)
val_cases, _ = train_test_split(temp_cases, test_size=0.5, random_state=42)

X_train, y_train = build_dataset(train_cases, PATCH_SIZE, MAX_VOXELS_PER_CASE, balance=True)
X_val, y_val = build_dataset(val_cases, PATCH_SIZE, MAX_VOXELS_PER_CASE, balance=True)

print("Training samples:", X_train.shape)
print("Validation samples:", X_val.shape)

n_features = X_train.shape[1]
print("Number of features:", n_features)

# -------------------------------------------------------------
# PyTorch Model
# -------------------------------------------------------------
class PatchMLP(nn.Module):
    def __init__(self, in_features, hidden_sizes=(64,32)):
        super().__init__()
        layers = []
        last = in_features
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = PatchMLP(n_features, hidden_sizes=(64,32)).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

train_loader = DataLoader(TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
), batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32)
), batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            logits_all.append(logits.cpu().numpy())
            y_all.append(yb.numpy())

    logits_all = np.concatenate(logits_all)
    y_all = np.concatenate(y_all)

    probs = 1 / (1 + np.exp(-logits_all))
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    preds = (probs > 0.5).astype(int)

    return log_loss(y_all, probs), f1_score(y_all, preds)

# -------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------
train_loss_curve = []
val_loss_curve = []
val_dice_curve = []

best_val_dice = -1
best_state = None

for epoch in range(1, N_EPOCHS+1):
    model.train()
    batch_losses = []

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    train_loss = np.mean(batch_losses)
    val_loss, val_dice = evaluate(model, val_loader)

    train_loss_curve.append(train_loss)
    val_loss_curve.append(val_loss)
    val_dice_curve.append(val_dice)

    print(f"Epoch {epoch:03d} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Dice={val_dice:.4f}")

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        best_state = model.state_dict().copy()

print("Best Val Dice:", best_val_dice)

# -------------------------------------------------------------
# Save Model
# -------------------------------------------------------------
save_dir = os.path.join("/home-local/sadridm/Projects/Spleen/MLP_Fusion_Inference", "models")
os.makedirs(save_dir, exist_ok=True)

torch.save({
    "state_dict": best_state,
    "in_features": n_features,
    "hidden_sizes": (64,32),
}, os.path.join(save_dir, "mlp_torch_best.pt"))

print("✅ Saved PyTorch model!")

# -------------------------------------------------------------
# Plot Curves
# -------------------------------------------------------------
epochs = np.arange(1, N_EPOCHS+1)

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(epochs, train_loss_curve, label="Train Loss")
plt.plot(epochs, val_loss_curve, label="Val Loss")
plt.title("Training vs Validation Loss")
plt.grid(); plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, val_dice_curve, label="Val Dice", color="green")
plt.title("Validation Dice Over Epochs")
plt.grid(); plt.legend()

plt.tight_layout()
plt.show()
