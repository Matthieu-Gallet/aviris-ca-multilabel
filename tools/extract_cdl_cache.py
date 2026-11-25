"""Extract the CDL counts of the release windows from the candidate-cell cache of the original
build (``cand_fine.npz``: cell, counts (n_cells, 14 years 2008-2021, 40 groups), groups).

Usage:
    python tools/extract_cdl_cache.py cand_fine.npz src/aviris_ml/data/windows.csv out.npz
"""
import csv
import sys

import numpy as np

from aviris_ml import config

src, windows_csv, out = sys.argv[1:4]
z = np.load(src)
if list(z["groups"]) != config.GROUP_NAMES:
    raise SystemExit("group list differs from config.CDL_GROUPS")
wid = np.array(sorted(int(r["window_id"]) for r in csv.DictReader(open(windows_csv))))
idx = np.searchsorted(z["cell"], wid)
if not (z["cell"][idx] == wid).all():
    raise SystemExit("some windows are missing in the cache")
np.savez_compressed(out, window_id=wid, counts=z["counts"][idx].astype(np.uint16), groups=np.array(config.GROUP_NAMES),
                    years=np.array(config.CDL_YEARS))
print(f"{len(wid)} windows written to {out}")
