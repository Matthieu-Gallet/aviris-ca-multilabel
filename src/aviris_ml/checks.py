"""Release checks: label statistics, co-occurrence, fold balance and leakage, value ranges,
NaN rate, spike statistics, consistency with the source dataset and with the shards."""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

from . import config, splits

log = logging.getLogger(__name__)


def _source_consistency(ctx, wid: np.ndarray, qz, lab) -> dict:
    """Comparison with an existing MIX folder (QC mask, per-date proportions, window proportions)."""
    out = {}
    src = ctx.from_existing
    if not src:
        return out
    m = os.path.join(src, "meta", "date_mask.npz")
    if os.path.exists(m):
        z = np.load(m)
        if np.array_equal(z["window_id"], wid):
            out["qc_ok_mismatches_vs_source"] = int(((qz["qc_ok"] > 0) != z["ok"]).sum())
    p = os.path.join(src, "meta", "props_by_year.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        if np.array_equal(z["window_id"], wid):
            acq = qz["qc_valid"] > 0
            d = np.abs(np.nan_to_num(lab["props_date"]) - z["props_dates"])[acq]
            out["props_date_max_abs_diff_vs_source"] = float(d.max())
    lc = os.path.join(src, "meta", "labels.csv")
    if os.path.exists(lc):
        L = pd.read_csv(lc)
        if np.array_equal(L.window_id.to_numpy(), wid):
            fr = [g for g, _ in config.CLASSES] + ["autre"]
            P = L[["p_" + c for c in fr]].to_numpy()
            lw = lab["labels_window"]
            alt = (P >= config.PRESENCE_THRESHOLD).astype(int)
            out["window_props_vs_source_mean_over_flight_years"] = dict(
                mean_abs_diff=float(np.abs(lab["props_window"] - P).mean()), max_abs_diff=float(np.abs(lab["props_window"] - P).max()),
                windows_with_different_labels=int((lw != alt).any(1).sum()), label_flips=int((lw != alt).sum()))
            out["source_fold5_vs_new_fold_agreement"] = float((L.fold5.to_numpy() == lab["fold"]).mean())
    return out


def run_checks(ctx, write_card: bool = True) -> dict:
    R = ctx.release
    lab = dict(np.load(ctx.p("labels.npz")))
    qz = dict(np.load(ctx.p("qc_stats.npz")))
    bt = pd.read_csv(os.path.join(R, "bands.csv")) if os.path.exists(os.path.join(R, "bands.csv")) else pd.read_csv(ctx.p("bands.csv"))
    sel = json.load(open(ctx.p("bands_selection.json")))
    wid = lab["window_id"]
    CL = config.CLASS_NAMES
    lw, ld, fold = lab["labels_window"], lab["labels_date"], lab["fold"].astype(int)
    acq, ok = qz["qc_valid"] > 0, qz["qc_ok"] > 0
    res: dict = {"n_windows": int(len(wid)), "n_window_dates": int(acq.size), "n_acquired": int(acq.sum()), "n_qc_ok": int(ok.sum())}

    # ---------------------------------------------------------------- labels
    npw = (lw == 1).sum(1)
    co = (lw.T.astype(int) @ lw.astype(int))
    union = (ld == 1).any(1)
    inter = np.where(acq[..., None], ld == 1, True).all(1)
    res["labels"] = dict(
        threshold=config.PRESENCE_THRESHOLD,
        labels_per_window={int(k): int((npw == k).sum()) for k in range(npw.max() + 1)},
        mean_labels_per_window=float(npw.mean()), windows_without_label=int((npw == 0).sum()),
        window_positives={c: int(lw[:, k].sum()) for k, c in enumerate(CL)},
        date_positives={c: int((ld[:, :, k] == 1).sum()) for k, c in enumerate(CL)},
        date_positives_qc_ok={c: int(((ld[:, :, k] == 1) & ok).sum()) for k, c in enumerate(CL)},
        labels_per_acquired_date={int(k): int(v) for k, v in zip(*np.unique((ld == 1).sum(-1)[acq], return_counts=True))},
        cooccurrence=dict(classes=CL, matrix=co.tolist()),
        windows_with_date_label_changes=int((union != inter).any(1).sum()),
        window_label_vs_union_of_dates=dict(windows_differing=int((union.astype(int) != lw).any(1).sum()), flips=int((union.astype(int) != lw).sum())),
        dominant_share_median=float(np.median(np.nanmax(lab["props_window"], 1))),
    )
    # ---------------------------------------------------------------- folds
    xy = np.array([[(config.cell_bounds(w)[0] + 300.0), (config.cell_bounds(w)[1] + 300.0)] for w in wid])
    fc = splits.check_folds(wid, fold, xy)
    per = {c: [int(lw[fold == f, k].sum()) for f in range(config.N_FOLDS)] for k, c in enumerate(CL)}
    rel = {c: float((max(v) - min(v)) / max(np.mean(v), 1)) for c, v in per.items()}
    fc.update(class_counts_per_fold=per, class_relative_range=rel, max_relative_range=float(max(rel.values())))
    res["folds"] = fc
    # ---------------------------------------------------------------- values
    keep = bt["kept"].to_numpy() > 0
    H, E = qz["hist_ok"], qz["hist_edges"]
    Hk = H[keep].sum(0)
    tot = Hk.sum()
    cdf = np.cumsum(Hk) / tot

    def q(p):
        i = int(np.searchsorted(cdf, p))
        return float(E[min(max(i - 1, 0), len(E) - 1)]) if 0 < i < len(E) + 1 else (float("-inf") if i == 0 else float("inf"))

    vmin = np.where(ok[..., None], qz["vmin"], np.nan)[..., keep]
    vmax = np.where(ok[..., None], qz["vmax"], np.nan)[..., keep]
    of = qz["qc_outlier_frac"]
    res["values_kept_bands_qc_ok"] = dict(
        n_values=int(tot), frac_lt_0=float(Hk[: np.searchsorted(E, 0.0, side="right")].sum() / tot),
        frac_gt_1=float(Hk[np.searchsorted(E, 1.0, side="right"):].sum() / tot), frac_lt_m0_2=float(Hk[0] / tot), frac_ge_2=float(Hk[-1] / tot),
        p001=q(0.001), p01=q(0.01), p50=q(0.5), p99=q(0.99), p999=q(0.999), min=float(np.nanmin(vmin)), max=float(np.nanmax(vmax)),
        window_dates_with_value_outside_m0_2_2=int(((vmin < -0.2) | (vmax > 2.0)).any(-1).sum()),
        outlier_frac_qc_ok=dict(mean=float(np.nanmean(of[ok])), frac_dates_gt0=float((of[ok] > 0).mean()), frac_dates_gt_1pct=float((of[ok] > 0.01).mean())))
    nb = qz["nanfrac_band"].astype("float32")
    res["nan"] = dict(
        pixel_nan_rate_acquired_kept_bands=float(nb[acq][:, keep].mean()), pixel_nan_rate_qc_ok_kept_bands=float(nb[ok][:, keep].mean()),
        nan_rate_incl_padding=float((np.where(acq[..., None], nb, 1.0)[..., keep]).mean()),
        windows_lt14_dates=int((acq.sum(1) < config.N_DATES).sum()),
        qc_ok_dates_per_window={"median": float(np.median(ok.sum(1))), "n14": int((ok.sum(1) == 14).sum()), "ge12": int((ok.sum(1) >= 12).sum()), "lt5": int((ok.sum(1) < 5).sum())},
        qc_reasons={k: int((qz["qc_" + k][acq] > 0).sum()) for k in ("nan_high", "snow", "bright", "zero")})
    # ---------------------------------------------------------------- spikes
    b0 = bt["kept_base"].to_numpy() > 0
    res["spikes"] = dict(selection=sel,
                         median_rate_base_166=float(np.nanmedian(bt.spike_rate_base[b0])), max_rate_base_166=float(np.nanmax(bt.spike_rate_base[b0])),
                         median_rate_final=float(np.nanmedian(bt.spike_rate_final[keep])), max_rate_final=float(np.nanmax(bt.spike_rate_final[keep])),
                         worst_final=[(float(bt.wavelength_nm[i]), float(bt.spike_rate_final[i])) for i in np.argsort(-bt.spike_rate_final.fillna(-1).to_numpy())[:8]],
                         gross_median_base=float(np.nanmedian(bt.gross_spike_rate_base[b0])), gross_max_base=float(np.nanmax(bt.gross_spike_rate_base[b0])),
                         gross_max_final=float(np.nanmax(bt.gross_spike_rate_final[keep])),
                         max_rate_final_recomputed=float(np.nanmax(bt.spike_rate_final_recomputed[keep])),
                         worst_final_recomputed=[(float(bt.wavelength_nm[i]), float(bt.spike_rate_final_recomputed[i])) for i in np.argsort(-bt.spike_rate_final_recomputed.fillna(-1).to_numpy())[:6]],
                         top_outside_m0_2_2_kept=[(float(bt.wavelength_nm[i]), float(bt["frac_outside_m0.2_2_qc_ok"][i])) for i in np.argsort(-np.where(keep, bt["frac_outside_m0.2_2_qc_ok"].to_numpy(), -1))[:8]])
    res["source_consistency"] = _source_consistency(ctx, wid, qz, lab)
    # ---------------------------------------------------------------- shards
    sh = os.path.join(R, "data")
    if os.path.isdir(sh):
        import pyarrow.parquet as pq

        ids, info = [], []
        for f in sorted(os.listdir(sh)):
            if not f.endswith(".parquet"):
                continue
            pf = pq.ParquetFile(os.path.join(sh, f))
            t = pf.read(columns=["window_id", "fold"])
            ids += t.column("window_id").to_pylist()
            info.append(dict(file=f, rows=pf.metadata.num_rows, row_groups=pf.metadata.num_row_groups, gb=round(os.path.getsize(os.path.join(sh, f)) / 1e9, 3),
                             folds=sorted(set(t.column("fold").to_pylist()))))
        res["shards"] = dict(files=info, total_rows=len(ids), unique_windows=len(set(ids)), all_windows_present=bool(set(ids) == set(wid.tolist())),
                             max_gb=max(i["gb"] for i in info) if info else 0, folds_consistent=all(len(i["folds"]) == 1 for i in info))
    os.makedirs(os.path.join(R, "metadata"), exist_ok=True)
    with open(os.path.join(R, "metadata", "checks.json"), "w") as f:
        json.dump(res, f, indent=1)
    log.info("checks written")
    if write_card and os.path.exists(os.path.join(R, "metadata", "windows.csv")):
        from .card import write_card as wc

        wc(ctx, R, n_bands=int(keep.sum()) if not ctx.all_bands else config.N_BANDS_RAW)
    return res
