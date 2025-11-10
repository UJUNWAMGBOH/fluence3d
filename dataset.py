"""
dataset.py
-----------
Dataset and preprocessing utilities for 3D CT + Contour → 2D Fluence prediction.
"""

import os, glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ----------------------- Utility functions -----------------------
def normalize(x, vmin, vmax):
    return (x - vmin) / (vmax - vmin + 1e-8)

def upsample_chw_numpy(arr_chw, H, W):
    t = torch.from_numpy(arr_chw[None]).float()
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t[0].cpu().numpy()

def ensure_D1HW(vol):
    vol = np.asarray(vol)
    if vol.ndim == 4:
        if vol.shape[1] == 1: return vol
        if vol.shape[-1] == 1: return np.moveaxis(vol, -1, 1)
        if vol.shape[0] == 1: return np.moveaxis(vol, 0, 1)
        return np.moveaxis(vol, -1, 1)
    elif vol.ndim == 3:
        if 4 <= vol.shape[0] <= 4096:
            return vol[:, None, ...]
        else:
            vol = np.moveaxis(vol, -1, 0)
            return vol[:, None, ...]
    else:
        raise ValueError(f"Unsupported volume shape: {vol.shape}")

def resize_D1HW_to_DHW(vol_D1HW, D, H, W):
    t = torch.from_numpy(vol_D1HW.transpose(1, 0, 2, 3)[None]).float()
    t = F.interpolate(t, size=(D, H, W), mode="trilinear", align_corners=False)
    return t[0, 0].cpu().numpy()

# ----------------------- Dataset -----------------------
class CTContour3DToFluDataset(Dataset):
    """
    For each patient:
      X: (2, H, W, D)  -> [CT, Contour]
      Y: (9, H, W)     -> fluence maps
    """
    def __init__(self, root, task, ct_path, contour_path, flu_path,
                 H=128, W=128, D=64,
                 min_ct=-1000, max_ct=3068.43,
                 min_contour=0, max_contour=3,
                 min_flu=0, max_flu=0.1):
        self.root=root; self.task=task
        self.ct_path=ct_path; self.contour_path=contour_path; self.flu_path=flu_path
        self.H=H; self.W=W; self.D=D
        self.min_ct=min_ct; self.max_ct=max_ct
        self.min_contour=min_contour; self.max_contour=max_contour
        self.min_flu=min_flu; self.max_flu=max_flu
        self.items=[]
        self._build()

    def _build(self):
        ct_files = glob.glob(self.ct_path.format(root=self.root, task=self.task, id="*"))
        for ct_fp in ct_files:
            pid = os.path.basename(ct_fp).split("_")[0]
            cnt_fp = self.contour_path.format(root=self.root, task=self.task, id=pid)
            flu_fp = self.flu_path.format(root=self.root, task=self.task, id=pid)
            if not (os.path.exists(cnt_fp) and os.path.exists(flu_fp)): continue

            ct  = ensure_D1HW(np.load(ct_fp).astype(np.float32))
            cnt = ensure_D1HW(np.load(cnt_fp).astype(np.float32))
            flu = np.load(flu_fp).astype(np.float32)
            if flu.ndim == 4 and flu.shape[-1] == 1:
                flu = np.squeeze(flu, -1)

            ct_dhw  = resize_D1HW_to_DHW(ct,  self.D, self.H, self.W)
            cnt_dhw = resize_D1HW_to_DHW(cnt, self.D, self.H, self.W)
            if (flu.shape[1]!=self.H) or (flu.shape[2]!=self.W):
                flu = upsample_chw_numpy(flu, self.H, self.W)

            ct_n   = np.clip(normalize(ct_dhw,  self.min_ct,      self.max_ct), 0,1)
            cnt_n  = np.clip(normalize(cnt_dhw, self.min_contour, self.max_contour), 0,1)
            flu_n  = np.clip(normalize(flu,     self.min_flu,     self.max_flu), 0,1)

            x = np.stack([
                np.moveaxis(ct_n,  0, -1),
                np.moveaxis(cnt_n, 0, -1)
            ], axis=0)
            self.items.append((x.astype(np.float32), flu_n.astype(np.float32)))

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        x, y = self.items[idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
