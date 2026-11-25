"""Loading example for the AVIRIS-C California multilabel dataset.

Usage:
    python load_example.py [DATASET]     # DATASET = local folder of the release or "<user>/<name>" on the Hub

Requires: datasets, numpy, pandas (torch optional).
"""
import os
import sys

import numpy as np
from datasets import load_dataset

src = sys.argv[1] if len(sys.argv) > 1 else "."
if os.path.isdir(src):
    src = os.path.abspath(src)  # local release folder

# 1. Metadata only (fast: no cubes) -----------------------------------------------------
windows = load_dataset(src, "windows", split="train").to_pandas()
print(windows[["window_id", "lat", "lon", "zone", "fold", "n_ok_dates", "n_labels"]].head())
classes = [c[len("label_"):] for c in windows.columns if c.startswith("label_")]
print("classes:", classes)

# 2. One fold, streamed (no full download) ---------------------------------------------------
stream = load_dataset(src, split="fold_0", streaming=True)
s = next(iter(stream))
cube = np.asarray(s["cube"], dtype=np.float32)  # (14, B, 32, 32), NaN = no data
ok = np.asarray(s["qc_ok"], dtype=bool)
print("window", s["window_id"], "cube", cube.shape, "QC-ok dates", ok.sum(), "dates", [d for d in s["dates"] if d][:3], "...")
print("window labels:", [c for c, y in zip(classes, s["labels_window"]) if y == 1])

# 3. Training array for one fold: valid dates only, NaN -> 0, mask kept ----------------------
ds = load_dataset(src, split="fold_1").with_format("numpy")
sub = ds.select(range(min(8, len(ds))))
X = np.stack(sub["cube"]).astype(np.float32)          # (n, 14, B, 32, 32)
M = np.stack(sub["qc_ok"]).astype(bool)                # (n, 14)
Y = np.stack(sub["labels_window"]).astype(np.int64)    # (n, 13) window multilabel
Yt = np.stack(sub["labels_date"]).astype(np.int64)     # (n, 14, 13), -1 = padded date
X = np.clip(np.nan_to_num(X, nan=0.0), -0.05, 1.5)     # recommended clipping of rare outliers
X[~M] = 0.0
print("batch", X.shape, M.shape, Y.shape, Yt.shape)

try:  # optional: PyTorch tensors
    import torch

    xb, mb, yb = torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(Y).float()
    print("torch", tuple(xb.shape), tuple(mb.shape), tuple(yb.shape))
except ImportError:
    pass
