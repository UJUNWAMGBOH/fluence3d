"""
evaluate.py
------------
Generate predictions and visualizations for test patients.
"""

import os, glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset import normalize, ensure_D1HW, resize_D1HW_to_DHW, upsample_chw_numpy
from train import Fluence3DModel
from scipy.optimize import linear_sum_assignment

def align_pred_to_gt(pred, gt):
    C = pred.shape[0]
    cost = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            cost[i,j] = np.mean((pred[j]-gt[i])**2)
    rows, cols = linear_sum_assignment(cost)
    return pred[cols]

@torch.no_grad()
def evaluate_all(model, device, root, ct_path, cnt_path, flu_path,
                 H=128, W=128, D=64, out_dir="./results"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval().to(device)
    ct_files = glob.glob(ct_path.format(root=root, task="test", id="*"))
    pids = sorted([os.path.basename(f).split("_")[0] for f in ct_files])
    for pid in pids:
        ct_fp  = ct_path.format(root=root, task="test", id=pid)
        cnt_fp = cnt_path.format(root=root, task="test", id=pid)
        flu_fp = flu_path.format(root=root, task="test", id=pid)
        if not (os.path.exists(ct_fp) and os.path.exists(cnt_fp) and os.path.exists(flu_fp)):
            print(f"Skipping {pid}, missing data"); continue

        ct  = ensure_D1HW(np.load(ct_fp).astype(np.float32))
        cnt = ensure_D1HW(np.load(cnt_fp).astype(np.float32))
        flu = np.load(flu_fp).astype(np.float32)
        if flu.ndim == 4 and flu.shape[-1] == 1:
            flu = np.squeeze(flu, -1)

        ct = resize_D1HW_to_DHW(ct, D, H, W)
        cnt = resize_D1HW_to_DHW(cnt, D, H, W)
        if (flu.shape[1]!=H) or (flu.shape[2]!=W):
            flu = upsample_chw_numpy(flu, H, W)

        x = np.stack([np.moveaxis(ct,0,-1), np.moveaxis(cnt,0,-1)], axis=0)[None]
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        pred = model(x_t).sigmoid()[0].cpu().numpy()
        pred = align_pred_to_gt(pred, flu)

        save_dir = os.path.join(out_dir, pid)
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(9, 3, figsize=(12, 30))
        for i in range(9):
            a0,a1,a2 = axes[i]
            diff = np.abs(pred[i]-flu[i])
            a0.imshow(flu[i], cmap="jet"); a0.set_title("GT"); a0.axis("off")
            a1.imshow(pred[i], cmap="jet"); a1.set_title("Pred"); a1.axis("off")
            a2.imshow(diff, cmap="hot");   a2.set_title("Diff"); a2.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "comparison.png"), dpi=150)
        plt.close(fig)
        print(f"Saved results for {pid}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Fluence3DModel()
    model.load_state_dict(torch.load("./checkpoints/best.pth", map_location=device))
    evaluate_all(model, device,
                 root="/home/uj/Prostate/data",
                 ct_path="{root}/{task}/ct/{id}_ct.npy",
                 cnt_path="{root}/{task}/contour/{id}_contoursCT.npy",
                 flu_path="{root}/{task}/fluences/{id}_fluences.npy",
                 out_dir="./eval_results")
