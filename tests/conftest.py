import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

from aviris_ml import config

WL = np.round(np.linspace(365.9, 2496.2, 224), 2)


def header_text(samples=40, lines=50, bands=224, dtype=4, interleave="bil", rotation=0.0, wl=WL):
    rot = f", rotation={rotation:.4f}" if rotation else ""
    w = " , ".join(f"{x:.2f}" for x in wl)
    f = " , ".join("10.0" for _ in wl)
    return (f"ENVI\nsamples = {samples}\nlines = {lines}\nbands = {bands}\nheader offset = 0\nfile type = ENVI Standard\n"
            f"data type = {dtype}\ninterleave = {interleave}\nbyte order = 0\n"
            f"map info = {{ UTM , 1 , 1 , 500000.000 , 4000000.000 , 15.000 , 15.000 , 10 , North , WGS-84 , units=Meters{rot} }}\n"
            f"wavelength = {{ {w} }}\nfwhm = {{ {f} }}\n")


def synthetic_spectrum(n=224, wl=WL):
    s = 0.05 + 0.3 / (1 + np.exp(-(wl - 720) / 15))  # vegetation-like red edge
    return s.astype("float32")


@pytest.fixture
def fake_mix(tmp_path):
    """A tiny AVIRIS-C_CA_TS14_MIX-like folder: 6 windows in 2 blocks, 14 dates, 224 bands."""
    rng = np.random.default_rng(0)
    root = tmp_path / "MIX"
    (root / "windows").mkdir(parents=True)
    (root / "meta").mkdir()
    hdr = tmp_path / "hdr"
    hdr.mkdir()
    (hdr / "f130410t01p00r11.hdr").write_text(header_text())
    ids = [36000057, 36000058, 36000100, 36000140, 40480136, 40480200]  # 5 blocks of 10 km
    dates = [f"{y}-0{m}-1{k}" for y, m, k in [(2013, 4, 0), (2013, 6, 1), (2014, 5, 2), (2014, 5, 3), (2014, 9, 4), (2015, 2, 5), (2015, 4, 6),
                                                 (2015, 6, 7), (2015, 6, 8), (2015, 9, 9), (2016, 6, 0), (2018, 8, 1), (2018, 8, 2), (2020, 9, 3)]]
    rows = []
    for n, w in enumerate(ids):
        cube = np.empty((14, 224, 32, 32), "float16")
        for t in range(14):
            s = synthetic_spectrum() * (1 + 0.05 * t / 14)
            cube[t] = (s[:, None, None] + 0.002 * rng.standard_normal((224, 32, 32))).astype("float16")
        cube[3, 145] += 5.0  # spike at band 145 on one date
        d = list(dates)
        fl = ["f130410t01p00r11"] * 14
        nf = np.zeros(14, "float32")
        if n == 5:  # padded last date
            cube[13] = np.nan
            d[13], fl[13], nf[13] = "", "", 1.0
        if n == 4:  # snowy date: very bright visible, dark SWIR
            cube[2, :] = 0.05
            cube[2, :60] = 0.8
        np.savez_compressed(root / "windows" / f"{w}_test.npz", cube=cube, dates=np.array(d), flights=np.array(fl), nan_frac=nf,
                            window_id=w, dominant="test", failures=np.array([]))
        x0, y0, x1, y1 = config.cell_bounds(w)
        rows.append(dict(window_id=w, lat=38.8 if w < 4e7 else 37.79, lon=-123.37 if w < 4e7 else -122.41, x_min=x0, y_min=y0, x_max=x1,
                         y_max=y1, dates=" ".join(x for x in d if x), dates_all=" ".join(dates)))
    import csv

    wcsv = tmp_path / "windows.csv"
    with open(wcsv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    counts = np.zeros((len(ids), 14, len(config.GROUP_NAMES)), np.uint16)
    gi = {g: i for i, g in enumerate(config.GROUP_NAMES)}
    for n in range(len(ids)):
        for y in range(14):
            counts[n, y, gi["arbustes"]] = 200 + 10 * (n % 3)
            counts[n, y, gi["foret_coniferes"]] = 150 - 10 * (n % 3)
            counts[n, y, gi["eau"]] = 30 + (y % 2) * 20  # 0.075 / 0.125: flips around 10 %
            counts[n, y, gi["luzerne"]] = 20
    cache = tmp_path / "cdl.npz"
    np.savez_compressed(cache, window_id=np.array(ids), counts=counts, groups=np.array(config.GROUP_NAMES), years=np.arange(2008, 2022))
    return dict(root=str(root), hdr=str(hdr), windows_csv=str(wcsv), cache=str(cache), ids=ids, tmp=tmp_path)

