"""Hugging Face release layout.

release/
  README.md                       dataset card (YAML header with configs)
  bands.csv  classes.csv          band table (224 rows, ``kept``) and class table with counts
  data/fold_<k>-<i>-of-<n>.parquet  one row = one window (14-date cube + labels), zstd
  metadata/windows.parquet|csv    one row per window (location, zone, block, fold, window labels)
  metadata/dates.parquet|csv      one row per (window, date): date, flight line, QC, per-date labels
  metadata/cdl_by_year.parquet    CDL group counts and class proportions per window and year 2008-2021
  metadata/flight_lines.csv       flight lines used (header info, ORNL URL)
  metadata/cdl_groups.csv         the 40 CDL groups, their codes and their class
  metadata/checks.json            build checks (written by ``aviris-ml check``)
  scripts/load_example.py  scripts/sanity_check.py
"""
from __future__ import annotations

import csv
import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from importlib import resources

import numpy as np
import pandas as pd

from . import config
from .envi import parse_header

log = logging.getLogger(__name__)
ROW_GROUP = 16
MAX_WINDOWS_PER_SHARD = 210


def features(n_bands: int):
    from datasets import Array2D, Array4D, Features, Sequence, Value

    T, K = config.N_DATES, config.N_CLASSES
    return Features({
        "window_id": Value("int64"), "fold": Value("int8"), "zone": Value("string"), "block10km": Value("int64"),
        "lat": Value("float32"), "lon": Value("float32"), "x_min": Value("float64"), "y_max": Value("float64"),
        "cube": Array4D(shape=(T, n_bands, config.NPIX, config.NPIX), dtype="float16"),
        "dates": Sequence(Value("string"), length=T), "acquired": Sequence(Value("bool"), length=T),
        "qc_ok": Sequence(Value("bool"), length=T), "flight_ids": Sequence(Value("string"), length=T),
        "nan_frac": Sequence(Value("float32"), length=T), "n_ok_dates": Value("int8"),
        "labels_date": Array2D(shape=(T, K), dtype="int8"), "props_date": Array2D(shape=(T, K), dtype="float32"),
        "labels_window": Sequence(Value("int8"), length=K), "props_window": Sequence(Value("float32"), length=K),
    })


def _write_shard(args) -> tuple[str, int, int]:
    path, recs, paths, band_idx = args
    import datasets
    import pyarrow.parquet as pq

    from .assemble import load_cube

    feat = features(len(band_idx))
    schema = feat.arrow_schema
    tmp = path + ".part"
    w = pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=6)
    for k in range(0, len(recs), ROW_GROUP):
        batch = []
        for r in recs[k:k + ROW_GROUP]:
            cube = load_cube(paths[r["window_id"]])["cube"][:, band_idx]
            batch.append(dict(r, cube=np.ascontiguousarray(cube, dtype=np.float16)))
        tbl = datasets.Dataset.from_dict({c: [b[c] for b in batch] for c in feat}, features=feat).data.table
        w.write_table(tbl, row_group_size=ROW_GROUP)
    w.close()
    os.replace(tmp, path)
    return path, len(recs), os.path.getsize(path)


def _header_info(dirs: list[str]) -> dict[str, dict]:
    out = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".hdr") and f[:-4] not in out:
                with open(os.path.join(d, f)) as fh:
                    h = parse_header(fh.read())
                out[f[:-4]] = dict(samples=h.samples, lines=h.lines, data_type=h.dtype, interleave=h.interleave,
                                   reflectance_scale=h.scale, rotation_deg=h.rotation, pixel_size_m=h.pixel_size)
    return out


def export_release(ctx) -> None:
    from .pipeline import cube_index, header_dirs

    R = ctx.release
    for d in ("data", "metadata", "scripts"):
        os.makedirs(os.path.join(R, d), exist_ok=True)
    lab = dict(np.load(ctx.p("labels.npz")))
    qz = dict(np.load(ctx.p("qc_stats.npz")))
    bt = pd.read_csv(ctx.p("bands.csv"))
    wid = lab["window_id"]
    assert (qz["window_id"] == wid).all()
    band_idx = np.arange(config.N_BANDS_RAW) if ctx.all_bands else np.nonzero(bt["kept"].to_numpy() > 0)[0]
    rows = {int(r["window_id"]): r for r in ctx.rows}
    CL = config.CLASS_NAMES
    T = config.N_DATES
    dates = qz["dates"].astype(str)
    flights = qz["flights"].astype(str)
    acq = qz["qc_valid"] > 0
    ok = qz["qc_ok"] > 0
    nanf = qz["qc_nan_frac"].astype("float32")
    fold = lab["fold"].astype(int)

    # ---------------------------------------------------------------- windows table
    W = []
    for n, w in enumerate(wid):
        r = rows[int(w)]
        x0, y0, x1, y1 = config.cell_bounds(int(w))
        lat, lon = float(r["lat"]), float(r["lon"])
        rec = dict(window_id=int(w), lat=lat, lon=lon, x_min=x0, y_min=y0, x_max=x1, y_max=y1, crs=config.GRID_CRS,
                   zone=config.zone_of(lat, lon), block10km=config.block_id(int(w)), fold=int(fold[n]),
                   n_dates=int(acq[n].sum()), n_ok_dates=int(ok[n].sum()), n_labels=int((lab["labels_window"][n] == 1).sum()))
        for k, c in enumerate(CL):
            rec["label_" + c] = int(lab["labels_window"][n, k])
        for k, c in enumerate(CL):
            rec["prop_" + c] = round(float(lab["props_window"][n, k]), 5)
        W.append(rec)
    Wdf = pd.DataFrame(W)
    Wdf.to_parquet(os.path.join(R, "metadata", "windows.parquet"), index=False)
    Wdf.to_csv(os.path.join(R, "metadata", "windows.csv"), index=False)

    # ---------------------------------------------------------------- dates table
    D = []
    for n, w in enumerate(wid):
        for t in range(T):
            rec = dict(window_id=int(w), t=t, date=dates[n, t], flight_id=flights[n, t], acquired=bool(acq[n, t]), qc_ok=bool(ok[n, t]),
                       cdl_year=int(dates[n, t][:4]) if dates[n, t] else -1, nan_frac=float(nanf[n, t]),
                       outlier_frac=round(float(qz["qc_outlier_frac"][n, t]), 5))
            for k in ("nan_high", "snow", "bright", "zero"):
                rec["qc_" + k] = int(qz["qc_" + k][n, t]) if acq[n, t] else 0
            for k in ("blue", "green", "nir", "swir"):
                rec["median_" + k] = round(float(qz["qc_" + k][n, t]), 5)
            for k, c in enumerate(CL):
                rec["label_" + c] = int(lab["labels_date"][n, t, k])
            for k, c in enumerate(CL):
                rec["prop_" + c] = round(float(lab["props_date"][n, t, k]), 5)
            D.append(rec)
    Ddf = pd.DataFrame(D)
    Ddf.to_parquet(os.path.join(R, "metadata", "dates.parquet"), index=False)
    Ddf.to_csv(os.path.join(R, "metadata", "dates.csv"), index=False)

    # ---------------------------------------------------------------- CDL by year
    cz = dict(np.load(ctx.p("cdl_counts.npz")))
    pos = {int(w): i for i, w in enumerate(cz["window_id"])}
    cnt = cz["counts"][[pos[int(w)] for w in wid]]
    years = [int(y) for y in cz["years"]]
    Y = []
    for n, w in enumerate(wid):
        for yi, y in enumerate(years):
            rec = dict(window_id=int(w), year=y, n_mapped=int(lab["mapped_year"][n, yi]))
            for g, name in enumerate(config.GROUP_NAMES):
                rec["count_" + config.GROUP_EN[name]] = int(cnt[n, yi, g])
            for k, c in enumerate(CL):
                rec["prop_" + c] = round(float(lab["props_year"][n, yi, k]), 5)
            Y.append(rec)
    pd.DataFrame(Y).to_parquet(os.path.join(R, "metadata", "cdl_by_year.parquet"), index=False)

    # ---------------------------------------------------------------- CDL groups, classes, bands
    legend = json.load(open(config.packaged_cdl_legend()))
    gci = config.group_to_class_index()
    with open(os.path.join(R, "metadata", "cdl_groups.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["group_id", "group", "class", "cdl_codes", "cdl_names"])
        for g, name in enumerate(config.GROUP_NAMES):
            codes = config.CDL_GROUPS[name]
            wr.writerow([g, config.GROUP_EN[name], CL[gci[g]], " ".join(map(str, codes)), "; ".join(legend.get(str(c), "?") for c in codes)])
    codes = config.class_cdl_codes()
    C = []
    lw = lab["labels_window"]
    ld = lab["labels_date"]
    for k, c in enumerate(CL):
        rec = dict(id=k, name=c, cdl_codes=" ".join(map(str, codes[k])),
                   cdl_groups=" ".join(config.GROUP_EN[g] for gi, g in enumerate(config.GROUP_NAMES) if gci[gi] == k))
        for f_ in range(config.N_FOLDS):
            rec[f"windows_fold_{f_}"] = int((lw[fold == f_, k] == 1).sum())
        rec["windows_total"] = int((lw[:, k] == 1).sum())
        rec["window_dates_total"] = int((ld[:, :, k] == 1).sum())
        rec["window_dates_qc_ok"] = int(((ld[:, :, k] == 1) & ok).sum())
        C.append(rec)
    pd.DataFrame(C).to_csv(os.path.join(R, "classes.csv"), index=False)
    bout = bt.copy()
    bout["release_index"] = -1
    bout.loc[band_idx, "release_index"] = np.arange(len(band_idx))
    bout.to_csv(os.path.join(R, "bands.csv"), index=False)

    # ---------------------------------------------------------------- flight lines
    hinfo = _header_info(header_dirs(ctx))
    urls = {}
    if os.path.exists(ctx.p("cmr_flights.json")):
        urls = {k: v["url"] for k, v in json.load(open(ctx.p("cmr_flights.json"))).items()}
    use: dict[str, list] = {}
    for n in range(len(wid)):
        for t in range(T):
            if flights[n, t]:
                use.setdefault(flights[n, t], []).append(dates[n, t])
    FL = []
    for fid in sorted(use):
        rec = dict(flight_id=fid, date=sorted(set(use[fid]))[0], n_window_dates=len(use[fid]), **hinfo.get(fid, {}), ornl_url=urls.get(fid, ""))
        FL.append(rec)
    pd.DataFrame(FL).to_csv(os.path.join(R, "metadata", "flight_lines.csv"), index=False)

    # ---------------------------------------------------------------- shards
    paths = cube_index(ctx)
    jobs = []
    for f_ in range(config.N_FOLDS):
        idx = np.nonzero(fold == f_)[0]
        if not len(idx):
            continue
        nsh = int(np.ceil(len(idx) / MAX_WINDOWS_PER_SHARD))
        for s, part in enumerate(np.array_split(idx, nsh)):
            recs = []
            for n in part:
                recs.append(dict(
                    window_id=int(wid[n]), fold=int(fold[n]), zone=W[n]["zone"], block10km=W[n]["block10km"],
                    lat=W[n]["lat"], lon=W[n]["lon"], x_min=W[n]["x_min"], y_max=W[n]["y_max"],
                    dates=list(dates[n]), acquired=acq[n].tolist(), qc_ok=ok[n].tolist(), flight_ids=list(flights[n]),
                    nan_frac=nanf[n].tolist(), n_ok_dates=int(ok[n].sum()), labels_date=lab["labels_date"][n],
                    props_date=lab["props_date"][n], labels_window=lab["labels_window"][n].tolist(),
                    props_window=lab["props_window"][n].tolist()))
            p = os.path.join(R, "data", f"fold_{f_}-{s:05d}-of-{nsh:05d}.parquet")
            if os.path.exists(p) and not ctx.force:
                log.info("reuse %s", os.path.basename(p))
                continue
            jobs.append((p, recs, paths, band_idx))
    with ProcessPoolExecutor(ctx.workers) as ex:
        for p, n, size in ex.map(_write_shard, jobs):
            log.info("export: %s %d windows %.2f GB", os.path.basename(p), n, size / 1e9)

    # ---------------------------------------------------------------- scripts + card
    src = resources.files("aviris_ml") / "release_scripts"
    for name in ("load_example.py", "sanity_check.py"):
        shutil.copy(str(src / name), os.path.join(R, "scripts", name))
    from .checks import run_checks

    run_checks(ctx)  # writes metadata/checks.json and the dataset card
    log.info("export: release written to %s", R)
