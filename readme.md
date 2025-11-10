# Fluence3D_SwinUNETR: CT + Contour → 2D Fluence

Predict 9-channel fluence maps from 3D CT and contour volumes using a 3D SwinUNETR.

---

## Requirements
```bash
pip install -r requirements.txt


data/
 ├── train/
 │   ├── ct/{id}_ct.npy
 │   ├── contour/{id}_contoursCT.npy
 │   └── fluences/{id}_fluences.npy
 ├── val/
 └── test/

CT / Contour: (D,H,W) or (D,1,H,W)
Fluence: (9,H,W) (or (9,H,W,1))

Train: python train.py
This will train the model and save checkpoints to:
./checkpoints/

Evaluate: python evaluate.py
This will write results to 
./eval_results/


To evaluate run this:
python evaluate.py
Writes results to ./eval_results/ with per-patient visualizations.

Files
dataset.py — dataset and some preprocessing
train.py — model + training loop
evaluate.py — prediction + visualization

