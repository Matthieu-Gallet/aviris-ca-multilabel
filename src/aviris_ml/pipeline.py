"""Stage orchestration. Every stage reads/writes the work directory, is idempotent (outputs
are written atomically and an existing output is reused unless ``force``) and resumable."""
from __future__ import annotations

import csv
import json
import logging
import os
import shutil

import numpy as np
import pandas as pd

from . import bands as bandsmod
from . import config, labels as labmod, splits

log = logging.getLogger(__name__)
STAGES = ["download", "assemble", "qc", "groundtruth", "labels", "export"]


class Context:
    def __init__(self, work: str, release: str, from_existing: str | None = None, windows_csv: str | None = None,
                 headers: str | None = None, cdl_cache: str | None = None, workers: int = 3, dl_workers: int = 2,
                 window_ids: list[int] | None = None, max_dates: int = config.N_DATES, force: bool = False,
                 keep_raw: bool = False, all_bands: bool = False):
        self.work = os.path.abspath(os.path.expanduser(work))
        self.release = os.path.abspath(os.path.expanduser(release))
        self.from_existing = from_existing
        self.headers = headers
        self.cdl_cache = cdl_cache
        self.workers = workers
        self.dl_workers = dl_workers
        self.max_dates = max_dates
        self.force = force
        self.keep_raw = keep_raw
        self.all_bands = all_bands
        os.makedirs(self.work, exist_ok=True)
        rows = list(csv.DictReader(open(windows_csv or config.packaged_windows_csv())))
        if window_ids:
            keep = {int(w) for w in window_ids}
            rows = [r for r in rows if int(r["window_id"]) in keep]
        self.rows = rows
        self.ids = np.array(sorted(int(r["window_id"]) for r in rows))

    def p(self, *a) -> str:
        return os.path.join(self.work, *a)

    def done(self, path: str) -> bool:
        if os.path.exists(path) and not self.force:
            log.info("reuse %s", os.path.relpath(path, self.work))
            return True
        return False


# ------------------------------------------------------------------------------------ download
def stage_download(ctx: Context) -> None:
    if ctx.from_existing:
        log.info("download: skipped (--from-existing uses the extracted cubes of %s)", ctx.from_existing)
        return
    from . import cmr
    from .download import Downloader

    flights = cmr.search_flights(cache=ctx.p("cmr_flights.json"))
    dl = Downloader(ctx.work, flights, workers=ctx.dl_workers)
    res = dl.run(ctx.rows, max_dates=ctx.max_dates)
    with open(ctx.p("download_summary.json"), "w") as f:
        json.dump(dict(windows=len(res), bytes_read_this_run=dl.bytes_read,
                       windows_with_failures=sum(1 for r in res if r["failures"]), results=res), f, indent=1)


# ------------------------------------------------------------------------------------ assemble
def stage_assemble(ctx: Context) -> None:
    idx_path = ctx.p("cubes_index.json")
    if ctx.from_existing:
        from .assemble import index_existing

        idx = index_existing(ctx.from_existing)
        idx = {int(w): idx[int(w)] for w in ctx.ids if int(w) in idx}
        missing = [int(w) for w in ctx.ids if int(w) not in idx]
        if missing:
            log.warning("assemble: %d windows missing in %s", len(missing), ctx.from_existing)
    else:
        from .assemble import assemble_window

        idx = {}
        for w in ctx.ids:
            out = assemble_window(ctx.p("raw"), int(w), ctx.p("cubes", f"{int(w)}.npz"), keep_raw=ctx.keep_raw)
            if out:
                idx[int(w)] = out
    with open(idx_path, "w") as f:
        json.dump({str(k): v for k, v in sorted(idx.items())}, f, indent=0)
    log.info("assemble: %d windows indexed", len(idx))


def cube_index(ctx: Context) -> dict[int, str]:
    with open(ctx.p("cubes_index.json")) as f:
        return {int(k): v for k, v in json.load(f).items()}


# ------------------------------------------------------------------------------------ qc + bands
def header_dirs(ctx: Context) -> list[str]:
    dirs = [d for d in (ctx.headers, ctx.p("hdr")) if d and os.path.isdir(d)]
    return dirs


def stage_qc(ctx: Context) -> None:
    from . import qc

    bt_path = ctx.p("bands_reference.csv")
    if not ctx.done(bt_path):
        hdrs = [h for d in header_dirs(ctx) for h in bandsmod.headers_in(d)]
        if hdrs:
            bt = bandsmod.header_band_table(hdrs)
        else:
            log.info("qc: no ENVI headers given, using the packaged band table")
            bt = pd.read_csv(config.packaged("bands_reference.csv"))
        bt.to_csv(bt_path, index=False)
    bt = pd.read_csv(bt_path)
    wl = bt["wavelength_nm"].to_numpy()
    m0 = bandsmod.base_mask(wl)
    log.info("qc: base mask keeps %d of %d bands", m0.sum(), len(m0))
    stats = ctx.p("qc_stats.npz")
    if ctx.force and os.path.exists(stats):
        os.remove(stats)
    qc.run_qc(cube_index(ctx), wl, m0, stats, workers=ctx.workers)
    # band selection from the spike statistic on QC-ok window-mean spectra
    z = np.load(stats)
    spec, okd = z["mean_spec"], z["qc_ok"] > 0
    mask, rate0, dropped = bandsmod.select_bands(spec, okd, m0)
    bt["kept_base"] = m0.astype(int)
    bt["kept"] = mask.astype(int)
    bt["spike_rate_base"] = np.round(rate0, 5)
    # "final": the same statistic (base-mask neighbours) restricted to the kept bands;
    # "recomputed": statistic recomputed on the final band set (new run edges next to removed bands)
    bt["spike_rate_final"] = np.round(np.where(mask, rate0, np.nan), 5)
    bt["spike_rate_final_recomputed"] = np.round(bandsmod.spike_rates(spec, okd, mask), 5)
    g0 = bandsmod.spike_rates(spec, okd, m0, config.SPIKE_REL_GROSS)
    bt["gross_spike_rate_base"] = np.round(g0, 5)
    bt["gross_spike_rate_final"] = np.round(np.where(mask, g0, np.nan), 5)
    nb = z["nanfrac_band"].astype("float32")
    valid = z["qc_valid"] > 0
    bt["nan_frac_valid_dates"] = np.round(nb[valid].mean(0), 5)
    H, E = z["hist_ok"], z["hist_edges"]
    tot = np.maximum(H.sum(1), 1)
    bt["frac_lt0_qc_ok"] = np.round(H[:, : np.searchsorted(E, 0.0, side="right")].sum(1) / tot, 6)
    bt["frac_gt1_qc_ok"] = np.round(H[:, np.searchsorted(E, 1.0, side="right"):].sum(1) / tot, 6)
    bt["frac_outside_m0.2_2_qc_ok"] = np.round((H[:, 0] + H[:, -1]) / tot, 6)
    bt.to_csv(ctx.p("bands.csv"), index=False)
    med = float(np.nanmedian(rate0))
    with open(ctx.p("bands_selection.json"), "w") as f:
        json.dump(dict(n_base=int(m0.sum()), n_final=int(mask.sum()), dropped=[int(b) for b in dropped],
                       dropped_wavelengths=[float(wl[b]) for b in dropped], dropped_rates=[float(rate0[b]) for b in dropped],
                       median_rate_base=med, threshold=max(config.SPIKE_RATE_FACTOR * med, config.SPIKE_RATE_MIN), n_pairs=int(okd.sum()),
                       spike_rel=config.SPIKE_REL, spike_floor=config.SPIKE_FLOOR, rate_factor=config.SPIKE_RATE_FACTOR), f, indent=1)
    log.info("qc: %d bands kept after spike screening (dropped %s nm)", mask.sum(), [round(float(wl[b])) for b in dropped])


# ------------------------------------------------------------------------------------ groundtruth
def stage_groundtruth(ctx: Context) -> None:
    from . import groundtruth as gt

    out = ctx.p("cdl_counts.npz")
    if ctx.done(out):
        return
    cache = ctx.cdl_cache or (config.packaged("cdl_counts_reference.npz") if ctx.from_existing else None)
    if cache:
        wid, cnt, years = gt.load_counts(cache)
        pos = {int(w): i for i, w in enumerate(wid)}
        miss = [int(w) for w in ctx.ids if int(w) not in pos]
        if miss:
            raise SystemExit(f"groundtruth: {len(miss)} windows missing in the CDL cache")
        counts = {int(w): cnt[pos[int(w)]] for w in ctx.ids}
        log.info("groundtruth: %d windows taken from cache %s", len(counts), cache)
    else:
        counts = gt.fetch_counts([r for r in ctx.rows], workers=ctx.workers)
        years = config.CDL_YEARS
        log.info("groundtruth: %d windows fetched from Planetary Computer", len(counts))
    gt.save_counts(out, counts, years)


# ------------------------------------------------------------------------------------ labels
def stage_labels(ctx: Context) -> None:
    from .groundtruth import load_counts

    out = ctx.p("labels.npz")
    if ctx.done(out):
        return
    z = dict(np.load(ctx.p("qc_stats.npz")))
    wid_q = z["window_id"]
    wid, cnt, years = load_counts(ctx.p("cdl_counts.npz"))
    if not set(wid_q.tolist()) <= set(wid.tolist()):
        raise SystemExit("labels: some windows have no CDL counts (run the groundtruth stage)")
    pos = {int(w): i for i, w in enumerate(wid)}
    cnt = cnt[[pos[int(w)] for w in wid_q]]
    py, tot = labmod.year_props(cnt)
    dates = z["dates"]
    yi = {y: i for i, y in enumerate(years)}
    bad = [(int(wid_q[n]), str(d)) for n in range(len(wid_q)) for d in dates[n]
           if str(d) and tot[n, yi[int(str(d)[:4])]] < config.CDL_MIN_MAPPED]
    if bad:
        raise SystemExit(f"labels: {len(bad)} (window, date) pairs whose CDL year has < {config.CDL_MIN_MAPPED} mapped pixels, "
                         f"e.g. {bad[:3]} (failed CDL reads? rerun groundtruth with --force)")
    pdate = labmod.date_props(py, dates, years)
    pwin = labmod.window_props(py, dates, years)
    ldate = labmod.multilabel(pdate)
    lwin = labmod.multilabel(pwin)
    folds = splits.assign_folds(wid_q, lwin)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, window_id=wid_q, years=np.array(years), props_year=py, mapped_year=tot.astype(np.int32),
                        props_date=pdate, labels_date=ldate, props_window=pwin, labels_window=lwin, fold=folds.astype(np.int8),
                        classes=np.array(config.CLASS_NAMES))
    os.replace(tmp, out)
    log.info("labels: %d windows, folds %s", len(wid_q), np.bincount(folds).tolist())


# ------------------------------------------------------------------------------------ export
def stage_export(ctx: Context) -> None:
    from . import export

    export.export_release(ctx)


def run(ctx: Context, stages: list[str]) -> None:
    fn = {"download": stage_download, "assemble": stage_assemble, "qc": stage_qc, "groundtruth": stage_groundtruth,
          "labels": stage_labels, "export": stage_export}
    for s in stages:
        log.info("==== stage %s", s)
        fn[s](ctx)


def copy_if_missing(src: str, dst: str) -> None:
    if not os.path.exists(dst):
        shutil.copy(src, dst)
