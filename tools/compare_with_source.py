"""Compare a (partial) network rebuild with an existing MIX folder.

Checks, for every window and date re-downloaded in WORK: same flight line, same NaN pattern and
identical float16 values as the source cube; and, if WORK/cdl_counts.npz exists, identical CDL
counts to a reference cache.

Usage:
    python tools/compare_with_source.py WORK MIX_DIR [--cdl-ref CDL_CACHE.npz] [--out result.json]
"""
import argparse
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("work")
ap.add_argument("mix")
ap.add_argument("--cdl-ref", default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
res = {"cubes": [], "cdl": []}
for p in sorted(glob.glob(os.path.join(a.work, "cubes", "*.npz"))):
    x = np.load(p)
    wid = int(x["window_id"])
    src = glob.glob(os.path.join(a.mix, "windows", f"{wid}_*.npz"))
    if not src:
        continue
    y = np.load(src[0])
    yd = [str(d) for d in y["dates"]]
    for t, d in enumerate(str(v) for v in x["dates"]):
        if not d or d not in yd:
            continue
        u = x["cube"][t].astype("float32")
        v = y["cube"][yd.index(d)].astype("float32")
        fin = np.isfinite(u) & np.isfinite(v)
        res["cubes"].append(dict(window_id=wid, date=d, flight=str(x["flights"][t]), same_flight=str(x["flights"][t]) == str(y["flights"][yd.index(d)]),
                                 same_nan_pattern=bool((np.isnan(u) == np.isnan(v)).all()), max_abs_diff=float(np.abs(u[fin] - v[fin]).max()) if fin.any() else None))
cp = os.path.join(a.work, "cdl_counts.npz")
if a.cdl_ref and os.path.exists(cp):
    c, r = np.load(cp), np.load(a.cdl_ref)
    pos = {int(w): i for i, w in enumerate(r["window_id"])}
    for i, w in enumerate(c["window_id"]):
        if int(w) in pos:
            d = np.abs(c["counts"][i].astype(int) - r["counts"][pos[int(w)]].astype(int)).sum()
            res["cdl"].append(dict(window_id=int(w), years=f"{int(c['years'].min())}-{int(c['years'].max())}", abs_count_diff=int(d)))
res["all_identical"] = all(e["same_flight"] and e["same_nan_pattern"] and e["max_abs_diff"] == 0 for e in res["cubes"]) and all(e["abs_count_diff"] == 0 for e in res["cdl"])
txt = json.dumps(res, indent=1)
print(txt)
if a.out:
    open(a.out, "w").write(txt)
