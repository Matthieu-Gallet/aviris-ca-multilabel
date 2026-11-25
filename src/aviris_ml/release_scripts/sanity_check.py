"""Sanity check of the release: shapes, label counts against classes.csv, NaN, and Ledoit-Wolf
spectral covariance per QC-ok date on a few samples (SPD test).

Usage:
    python sanity_check.py [DATASET] [N_SAMPLES]
Requires: datasets, numpy, pandas, scikit-learn.
"""
import os
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.covariance import ledoit_wolf

src = sys.argv[1] if len(sys.argv) > 1 else "."
if os.path.isdir(src):
    src = os.path.abspath(src)  # local release folder
n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 6
fail = []

windows = load_dataset(src, "windows", split="train").to_pandas()
classes = [c[len("label_"):] for c in windows.columns if c.startswith("label_")]
K = len(classes)
print(f"{len(windows)} windows, {K} classes, folds {windows.fold.value_counts().sort_index().tolist()}")
if windows.window_id.duplicated().any():
    fail.append("duplicated window ids")
if windows.groupby("block10km").fold.nunique().max() > 1:
    fail.append("a 10 km block spans several folds")

if os.path.exists(os.path.join(src, "classes.csv")):
    C = pd.read_csv(os.path.join(src, "classes.csv"))
    for k, c in enumerate(classes):
        n = int(windows["label_" + c].sum())
        if n != int(C.windows_total[k]):
            fail.append(f"class {c}: {n} windows != classes.csv {C.windows_total[k]}")
    print("window label counts match classes.csv:", not any("classes.csv" in f for f in fail))
    for f in range(5):
        got = [int(windows.loc[windows.fold == f, "label_" + c].sum()) for c in classes]
        if got != C[f"windows_fold_{f}"].tolist():
            fail.append(f"fold {f} label counts differ from classes.csv")

ds = load_dataset(src, split="fold_0", streaming=True)
spd, total, shrink = 0, 0, []
for i, s in enumerate(ds):
    if i >= n_samples:
        break
    cube = np.asarray(s["cube"], dtype=np.float64)
    T, B, H, W = cube.shape
    ok = np.asarray(s["qc_ok"], bool)
    ld = np.asarray(s["labels_date"])
    if ld.shape != (T, K) or len(s["labels_window"]) != K:
        fail.append(f"label shapes {ld.shape}")
    acq = np.asarray(s["acquired"], bool)
    if not (ld[~acq] == -1).all() or not (ld[acq] >= 0).all():
        fail.append("padded-date label convention violated")
    for t in np.nonzero(ok)[0]:
        X = np.clip(cube[t].reshape(B, -1), -0.05, 1.5)
        X = X[:, np.isfinite(X).all(0)]
        if X.shape[1] < 2 * B:
            continue
        Cv, a = ledoit_wolf(X.T)
        lam = np.linalg.eigvalsh(Cv)
        total += 1
        spd += int(lam[0] > 0)
        shrink.append(a)
    print(f"sample {s['window_id']}: cube {cube.shape}, ok dates {ok.sum()}, NaN {np.isnan(cube[ok]).mean():.4f}, "
          f"labels {[c for c, y in zip(classes, s['labels_window']) if y == 1]}")
print(f"Ledoit-Wolf covariances {B}x{B}: {spd}/{total} SPD, median shrinkage {np.median(shrink):.4f}")
if spd != total:
    fail.append("non-SPD Ledoit-Wolf covariance")
print("OK" if not fail else "FAILED: " + "; ".join(fail))
sys.exit(1 if fail else 0)
