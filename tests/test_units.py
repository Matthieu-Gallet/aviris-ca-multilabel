import numpy as np
import pytest

from aviris_ml import bands, config, envi, labels, qc, splits
from aviris_ml.download import spread
from aviris_ml.groundtruth import class_counts, counts_from_codes

from conftest import WL, header_text, synthetic_spectrum


# ------------------------------------------------------------------------------- config
def test_grid_and_blocks():
    x0, y0, x1, y1 = config.cell_bounds(36000057)
    assert (x1 - x0, y1 - y0) == (600.0, 600.0)
    gy, gx = divmod(36000057, 20000)
    assert x0 == config.X0 + 600 * gx and y1 == config.Y0 - 600 * gy
    assert config.block_id(36000057) == (gx // 17) * 100000 + gy // 17
    assert config.zone_of(34.4, -119.8) == "Z2_SantaBarbara" and config.zone_of(40.0, -120.0) == "none"


def test_class_mapping_complete_and_disjoint():
    gci = config.group_to_class_index()
    assert len(gci) == len(config.GROUP_NAMES) == 40
    assert sorted(set(gci)) == list(range(config.N_CLASSES))
    codes = config.class_cdl_codes()
    flat = [c for cc in codes for c in cc]
    assert len(flat) == len(set(flat))  # a CDL code maps to one class only
    assert set(config.GROUP_EN) == set(config.GROUP_NAMES) and len(set(config.GROUP_EN.values())) == 40
    lut = config.cdl_lut()
    assert lut[142] == config.GROUP_NAMES.index("foret_coniferes") and lut[0] == 255 and lut[81] == 255


# ------------------------------------------------------------------------------- envi
def test_parse_header_and_decode():
    h = envi.parse_header(header_text(samples=5, lines=3, bands=4, dtype=2, interleave="bil", rotation=28.0, wl=WL[:4]))
    assert (h.samples, h.lines, h.bands, h.dtype, h.interleave) == (5, 3, 4, 2, "bil")
    assert h.rotation == 28.0 and h.pixel_size == 15.0 and h.scale == 1e-4 and h.supported
    assert len(h.wavelength) == 4 and len(h.fwhm) == 4
    a = np.arange(2 * 4 * 5, dtype="<i2").reshape(2, 4, 5)  # BIL rows: (row, band, sample)
    out = envi.decode_rows(a.tobytes(), h, 2)
    assert out.shape == (2, 5, 4) and out[1, 3, 2] == a[1, 2, 3]
    h.interleave = "bip"
    b = np.arange(2 * 5 * 4, dtype="<i2")
    assert envi.decode_rows(b.tobytes(), h, 2)[1, 3, 2] == b.reshape(2, 5, 4)[1, 3, 2]


# ------------------------------------------------------------------------------- assemble
@pytest.mark.parametrize("rotation", [0.0, 30.0])
def test_to_window_constant_field_and_nodata(rotation):
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    from aviris_ml.assemble import to_window

    wid = 36000057
    x0, y0, x1, y1 = config.cell_bounds(wid)
    px = 15.0
    # rotated grid centred on the window, in EPSG:5070 directly
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    n = 80
    tr = Affine.translation(cx, cy) * Affine.rotation(rotation) * Affine.translation(-n * px / 2, n * px / 2) * Affine.scale(px, -px)
    block = np.full((n, n, 3), 2500, np.int16)  # reflectance 0.25 (x 10000)
    block[:5] = -9999  # no-data rows far from the window
    cube, nf = to_window(block, 1e-4, tr, CRS.from_epsg(5070), wid)
    assert cube.shape == (3, 32, 32) and cube.dtype == np.float16
    assert nf == 0.0
    assert np.allclose(cube.astype("float32"), 0.25, atol=1e-3)


def test_nodata_mask():
    from aviris_ml.assemble import nodata_mask

    a = np.ones((2, 2, 3), "float32")
    a[0, 0] = 0
    a[0, 1, 1] = -9999
    a[1, 0, 2] = np.nan
    assert nodata_mask(a).tolist() == [[True, True], [True, False]]


# ------------------------------------------------------------------------------- qc
def test_qc_rules():
    m0 = bands.base_mask(WL)
    qb = qc.qc_bands(WL)
    good = np.broadcast_to(synthetic_spectrum()[:, None, None], (224, 32, 32)).astype("float16")
    assert qc.qc_date(good, 0.0, m0, qb)["ok"] == 1
    assert qc.qc_date(good, 0.3, m0, qb)["ok"] == 0
    snow = np.full((224, 32, 32), 0.05, "float16")
    snow[WL < 1000] = 0.8
    r = qc.qc_date(snow, 0.0, m0, qb)
    assert r["snow"] == 1 and r["ok"] == 0
    zero = np.zeros((224, 32, 32), "float16")
    assert qc.qc_date(zero, 0.0, m0, qb)["zero"] == 1


# ------------------------------------------------------------------------------- bands
def test_base_mask_ranges():
    m = bands.base_mask(WL)
    for lo, hi in config.BASE_REMOVED_RANGES_NM:
        assert not m[(WL >= lo) & (WL <= hi)].any()
    assert m[(WL > 420) & (WL < 1300)].all()
    assert len(bands.runs(m)) == 3


def test_explicit_940_removal():
    w = np.array([918.4, 928.1, 937.8, 947.4, 957.1, 966.7])
    assert bands.explicit_removed(w).tolist() == [False, True, True, True, True, False]


def test_spike_detection_isolated_band():
    rng = np.random.default_rng(1)
    m0 = bands.base_mask(WL)
    N, T = 40, 14
    spec = np.broadcast_to(synthetic_spectrum(), (N, T, 224)).copy() + 0.001 * rng.standard_normal((N, T, 224))
    b = int(np.argmin(np.abs(WL - 1650)))
    assert m0[b]
    spec[: N // 2, 3, b] += 0.5  # spike on one date for half of the windows
    ok = np.ones((N, T), bool)
    f = bands.spike_flags(spec, ok, m0)
    assert f[: N // 2, 3, b].all()
    assert not f[: N // 2, 3, b - 1].any() and not f[: N // 2, 3, b + 1].any()  # neighbours not flagged
    mask, rate0, dropped = bands.select_bands(spec, ok, m0)
    assert dropped == [b] and mask.sum() == m0.sum() - 1
    # the red edge (a surface feature present on all dates) is not a spike
    assert rate0[np.argmin(np.abs(WL - 720))] == 0


# ------------------------------------------------------------------------------- groundtruth / labels
def test_counts_and_props():
    codes = np.full((20, 20), 142, np.uint8)
    codes[:2] = 64      # shrubland: 40 px
    codes[2:3] = 81     # clouds: ignored
    c = counts_from_codes(codes)
    assert c.sum() == 380 and c[config.GROUP_NAMES.index("arbustes")] == 40
    cc = class_counts(c[None, None])
    assert cc.shape == (1, 1, 13) and cc.sum() == 380
    p, tot = labels.year_props(c[None, None].astype(np.uint16))
    assert np.isclose(p[0, 0, config.CLASS_NAMES.index("shrubland")], 40 / 380)


def test_date_and_window_labels():
    years = [2013, 2014]
    py = np.array([[[0.05, 0.95], [0.20, 0.80]]], "float32")  # (1 window, 2 years, 2 classes)
    dates = np.array([["2013-04-01", "2013-05-01", "2014-06-01", ""]])
    pd_ = labels.date_props(py, dates, years)
    assert np.isnan(pd_[0, 3]).all() and np.allclose(pd_[0, 2], [0.2, 0.8])
    lab = labels.multilabel(pd_)
    assert lab[0, 3].tolist() == [-1, -1] and lab[0, 0].tolist() == [0, 1] and lab[0, 2].tolist() == [1, 1]
    pw = labels.window_props(py, dates, years)  # distinct years 2013, 2014 weighted once
    assert np.allclose(pw[0], [0.125, 0.875])
    assert labels.multilabel(pw)[0].tolist() == [1, 1]
    assert labels.multilabel(np.array([0.1], "float32")).tolist() == [1]  # threshold inclusive


# ------------------------------------------------------------------------------- splits / download
def test_folds_block_safe_and_deterministic():
    rng = np.random.default_rng(0)
    gy = rng.integers(0, 200, 600)
    gx = rng.integers(0, 200, 600)
    ids = np.unique(gy * 20000 + gx)
    lab = (rng.random((len(ids), 5)) < 0.3).astype(np.int8)
    f1 = splits.assign_folds(ids, lab)
    f2 = splits.assign_folds(ids, lab)
    assert (f1 == f2).all() and set(f1) == set(range(5))
    xy = np.array([[config.cell_bounds(w)[0], config.cell_bounds(w)[1]] for w in ids])
    chk = splits.check_folds(ids, f1, xy)
    assert chk["blocks_in_several_folds"] == 0 and chk["duplicated_windows"] == 0
    counts = np.bincount(f1)
    assert counts.max() - counts.min() <= 0.3 * counts.mean()


def test_spread_dates():
    out = spread(["2013-01-01", "2013-01-02", "2016-01-01", "2020-01-01"], ["2013-01-01"], 3)
    assert out == ["2013-01-01", "2020-01-01", "2016-01-01"]
