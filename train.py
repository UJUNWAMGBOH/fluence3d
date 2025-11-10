"""
train.py
---------
Train the 3D SwinUNETR model to predict 2D fluence maps from CT + Contour.
"""

import os, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from monai.networks.nets import SwinUNETR
from dataset import CTContour3DToFluDataset

# -----------------------
# Model definition
# -----------------------
class DepthPool(nn.Module):
    def forward(self, x): return x.mean(dim=-1)

class Fluence3DModel(nn.Module):
    def __init__(self, img_size=(128,128,64), feature_size=48):
        super().__init__()
        self.net = SwinUNETR(
            img_size=img_size,
            in_channels=2, out_channels=9,
            feature_size=feature_size,
            spatial_dims=3)
        self.pool = DepthPool()
    def forward(self, x):
        return self.pool(self.net(x))

# -----------------------
# Training helpers
# -----------------------
def mse_loss(a, b): return F.mse_loss(a, b)
def mae_loss(a, b): return F.l1_loss(a, b)
def mixed_loss(pred, tgt, a=0.7, b=0.3):
    return a*mse_loss(pred, tgt) + b*mae_loss(pred, tgt)

def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        pred = model(x).sigmoid()
        loss = mixed_loss(pred, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).sigmoid()
        total_loss += mixed_loss(pred, y).item()
    return total_loss / len(loader)

# -----------------------
# Main training routine
# -----------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = "/home/uj/Prostate/data"
    ct_path   = "{root}/{task}/ct/{id}_ct.npy"
    cnt_path  = "{root}/{task}/contour/{id}_contoursCT.npy"
    flu_path  = "{root}/{task}/fluences/{id}_fluences.npy"

    train_ds = CTContour3DToFluDataset(data_root, "train", ct_path, cnt_path, flu_path)
    val_ds   = CTContour3DToFluDataset(data_root, "val", ct_path, cnt_path, flu_path)
    dl_train = DataLoader(train_ds, batch_size=1, shuffle=True)
    dl_val   = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = Fluence3DModel().to(device)
    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val = float("inf")
    os.makedirs("./checkpoints", exist_ok=True)

    for epoch in range(1, 2):
        tr_loss = train_epoch(model, dl_train, opt, device)
        va_loss = val_epoch(model, dl_val, device)
        print(f"[Epoch {epoch}] Train={tr_loss:.6f}, Val={va_loss:.6f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), "./checkpoints/best.pth")
            print("  -> Saved best model")

    torch.save(model.state_dict(), "./checkpoints/last.pth")
    print("Training complete.")
