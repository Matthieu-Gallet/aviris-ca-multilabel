"""Command line: ``aviris-ml build --stage ...`` and ``aviris-ml check``."""
from __future__ import annotations

import argparse
import logging
import os
import sys

from . import config
from .pipeline import STAGES, Context, run


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="aviris-ml", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="run one or several build stages")
    b.add_argument("--stage", action="append", choices=STAGES + ["all"], required=True,
                   help="stage to run (repeatable); 'all' runs them in order")
    b.add_argument("--work", default="~/projets/aviris-ca-multilabel-work", help="work directory (caches, intermediates)")
    b.add_argument("--release", default="~/projets/aviris-ca-multilabel-release", help="output directory of the HF release")
    b.add_argument("--from-existing", default=None, metavar="MIX_DIR",
                   help="use the already extracted cubes of an existing AVIRIS-C_CA_TS14_MIX folder (read-only)")
    b.add_argument("--windows-csv", default=None, help=f"window list (default: packaged {os.path.basename(config.packaged_windows_csv())})")
    b.add_argument("--window-ids", default=None, help="comma-separated subset of window ids")
    b.add_argument("--limit", type=int, default=0, help="only the first N windows of the list")
    b.add_argument("--headers", default=None, help="directory of ENVI .hdr files of the flight lines (band table)")
    b.add_argument("--cdl-cache", default=None, help="npz of CDL group counts per window and year (skips the CDL download)")
    b.add_argument("--workers", type=int, default=3, help="CPU workers (QC pass, CDL reads)")
    b.add_argument("--dl-workers", type=int, default=2, help="concurrent HTTP range requests to ORNL")
    b.add_argument("--max-dates", type=int, default=config.N_DATES, help="dates per window to download (tests)")
    b.add_argument("--keep-raw", action="store_true", help="keep the native pixel blocks after assembly")
    b.add_argument("--all-bands", action="store_true", help="export the 224 raw bands instead of the cleaned subset")
    b.add_argument("--force", action="store_true", help="recompute outputs that already exist")
    c = sub.add_parser("check", help="compute the release checks (checks.json)")
    c.add_argument("--work", default="~/projets/aviris-ca-multilabel-work")
    c.add_argument("--release", default="~/projets/aviris-ca-multilabel-release")
    c.add_argument("--from-existing", default=None)
    c.add_argument("--all-bands", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    for noisy in ("urllib3", "rasterio", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)  # never echo request URLs (tokens)
    if a.cmd == "check":
        from .checks import run_checks

        ctx = Context(a.work, a.release, from_existing=a.from_existing, all_bands=a.all_bands)
        run_checks(ctx)
        return 0
    ids = [int(x) for x in a.window_ids.split(",")] if a.window_ids else None
    ctx = Context(a.work, a.release, from_existing=a.from_existing, windows_csv=a.windows_csv, headers=a.headers,
                  cdl_cache=a.cdl_cache, workers=a.workers, dl_workers=a.dl_workers, window_ids=ids, max_dates=a.max_dates,
                  force=a.force, keep_raw=a.keep_raw, all_bands=a.all_bands)
    if a.limit:
        ctx.rows = ctx.rows[: a.limit]
        ctx.ids = ctx.ids[: a.limit] if not ids else ctx.ids
        ctx.ids = sorted(int(r["window_id"]) for r in ctx.rows)
    stages = STAGES if "all" in a.stage else [s for s in STAGES if s in a.stage]
    run(ctx, stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
