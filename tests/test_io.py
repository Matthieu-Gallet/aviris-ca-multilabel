"""HTTP Range download against a local server, CDL window read, and a tiny end-to-end build."""
import http.server
import json
import os
import threading

import numpy as np
import pytest

from aviris_ml import config


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404)
            return
        data = open(path, "rb").read()
        rng = self.headers.get("Range")
        if rng:
            a, b = rng.split("=")[1].split("-")
            a, b = int(a), int(b)
            chunk = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        else:
            chunk = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)


@pytest.fixture
def server(tmp_path):
    d = tmp_path / "srv"
    d.mkdir()
    handler = lambda *a, **k: RangeHandler(*a, directory=str(d), **k)  # noqa: E731
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield d, f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_fetch_range_resume_and_errors(server, tmp_path):
    from aviris_ml.download import FatalHTTP, fetch

    d, url = server
    payload = np.arange(10000, dtype="<i2").tobytes()
    (d / "img.bin").write_bytes(payload)
    out = str(tmp_path / "part.bin")
    n = fetch(url + "/img.bin", out, (200, 399), expected=200)
    assert n == 200 and open(out, "rb").read() == payload[200:400]
    mtime = os.path.getmtime(out)
    assert fetch(url + "/img.bin", out, (200, 399), expected=200) == 200 and os.path.getmtime(out) == mtime  # reused
    with pytest.raises(IOError):
        fetch(url + "/img.bin", str(tmp_path / "bad.bin"), (0, 99), expected=50, retries=1)
    assert not os.path.exists(str(tmp_path / "bad.bin")) and not os.path.exists(str(tmp_path / "bad.bin.part"))
    with pytest.raises(FatalHTTP):
        fetch(url + "/missing.bin", str(tmp_path / "m.bin"), retries=3)


def test_read_window_codes_on_cdl_grid(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    from aviris_ml.groundtruth import counts_from_codes, read_window_codes

    wid = 36000057
    x0, y0, x1, y1 = config.cell_bounds(wid)
    ox, oy = x0 - 30 * 50, y1 + 30 * 40  # tile origin on the 30 m grid
    a = np.full((200, 200), 176, np.uint8)
    a[40:60, 50:70] = 142  # exactly the window
    p = str(tmp_path / "cdl.tif")
    with rasterio.open(p, "w", driver="GTiff", width=200, height=200, count=1, dtype="uint8", crs="EPSG:5070",
                       transform=from_origin(ox, oy, 30, 30)) as ds:
        ds.write(a, 1)
    with rasterio.open(p) as ds:
        codes = read_window_codes(ds, wid)
    assert codes.shape == (20, 20) and (codes == 142).all()
    assert counts_from_codes(codes)[config.GROUP_NAMES.index("foret_coniferes")] == 400


def test_end_to_end_from_existing(fake_mix, tmp_path):
    from aviris_ml.pipeline import STAGES, Context, run

    work, rel = str(tmp_path / "work"), str(tmp_path / "release")
    ctx = Context(work, rel, from_existing=fake_mix["root"], windows_csv=fake_mix["windows_csv"], headers=fake_mix["hdr"],
                  cdl_cache=fake_mix["cache"], workers=1)
    run(ctx, STAGES)
    run(ctx, STAGES)  # idempotent: second run reuses outputs
    sel = json.load(open(os.path.join(work, "bands_selection.json")))
    assert 145 in sel["dropped"]  # the planted spike band is removed
    import pandas as pd

    W = pd.read_csv(os.path.join(rel, "metadata", "windows.csv"))
    assert len(W) == 6 and W.groupby("block10km").fold.nunique().max() == 1
    D = pd.read_parquet(os.path.join(rel, "metadata", "dates.parquet"))
    assert len(D) == 6 * 14 and (~D.acquired).sum() == 1
    pad = D[~D.acquired].iloc[0]
    assert pad.label_water == -1 and not pad.qc_ok
    # water flips around 10 % between odd / even years: per-date labels vary
    d0 = D[(D.window_id == 36000057) & D.acquired]
    assert set(d0.label_water) == {0, 1}
    chk = json.load(open(os.path.join(rel, "metadata", "checks.json")))
    assert chk["shards"]["all_windows_present"] and chk["folds"]["blocks_in_several_folds"] == 0
    assert chk["source_consistency"] == {} or "qc_ok_mismatches_vs_source" not in chk["source_consistency"]
    from datasets import load_dataset

    cache = str(tmp_path / "hf_cache")
    ds = load_dataset(os.path.abspath(rel), split="fold_0", cache_dir=cache).with_format("numpy")
    s = ds[0]
    nb = sel["n_final"]
    assert s["cube"].shape == (14, nb, 32, 32) and s["labels_date"].shape == (14, 13)
    meta = load_dataset(os.path.abspath(rel), "windows", split="train", cache_dir=cache)
    assert len(meta) == 6
    readme = open(os.path.join(rel, "README.md")).read()
    assert readme.startswith("---\n") and "multi-label-image-classification" in readme
