"""Dataset card (README.md with Hugging Face YAML header) generated from the build outputs."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import config

PRETTY = {
    "almond": "Almonds", "shrubland": "Shrubland", "water": "Water (open water, aquaculture)",
    "evergreen_forest": "Evergreen forest", "mixed_forest": "Mixed forest", "grassland_pasture": "Grassland / pasture",
    "barren": "Barren", "developed_high": "Developed, high intensity", "developed_low": "Developed, low intensity",
    "developed_medium": "Developed, medium intensity", "developed_open": "Developed, open space", "grapes": "Grapes (vineyards)",
    "other": "Other: all remaining mapped CDL groups (crops, other orchards, deciduous forest, wetlands, snow/ice)",
}


def _ranges(wl: np.ndarray, keep: np.ndarray) -> list[str]:
    out, i = [], 0
    while i < len(keep):
        if not keep[i]:
            j = i
            while j + 1 < len(keep) and not keep[j + 1]:
                j += 1
            out.append(f"{wl[i]:.0f}" if i == j else f"{wl[i]:.0f}-{wl[j]:.0f}")
            i = j + 1
        else:
            i += 1
    return out


def write_card(ctx, R: str, n_bands: int) -> None:
    W = pd.read_csv(os.path.join(R, "metadata", "windows.csv"))
    D = pd.read_parquet(os.path.join(R, "metadata", "dates.parquet"))
    C = pd.read_csv(os.path.join(R, "classes.csv"))
    B = pd.read_csv(os.path.join(R, "bands.csv"))
    sel = json.load(open(ctx.p("bands_selection.json")))
    shards = sorted(f for f in os.listdir(os.path.join(R, "data")) if f.endswith(".parquet"))
    size_gb = sum(os.path.getsize(os.path.join(R, "data", f)) for f in shards) / 1e9
    N, T, K = len(W), config.N_DATES, config.N_CLASSES
    n_acq = int(D.acquired.sum())
    n_ok = int(D.qc_ok.sum())
    wl = B.wavelength_nm.to_numpy()
    base_removed = _ranges(wl, B.kept_base.to_numpy() > 0)
    extra = [f"{w:.0f}" for w in sorted(sel["dropped_wavelengths"])]
    qc_counts = {k: int((D[f"qc_{k}"] > 0).sum()) for k in ("nan_high", "snow", "bright", "zero")}
    nlab = W.n_labels.value_counts().sort_index()
    zones = W.zone.value_counts()
    folds = W.fold.value_counts().sort_index()
    span = D[D.acquired].date.agg(["min", "max"])
    yrs = pd.to_datetime(D[D.acquired].date).dt.year.value_counts().sort_index()
    chk_path = os.path.join(R, "metadata", "checks.json")
    chk = json.load(open(chk_path)) if os.path.exists(chk_path) else {}
    chk_vals, chk_spk = chk.get("values_kept_bands_qc_ok", {}), chk.get("spikes", {})
    leak = chk.get("folds", {}).get("nearest_other_fold_m", {})

    cls_rows = "\n".join(
        f"| {r['id']} | `{r['name']}` | {PRETTY[r['name']]} | {r['cdl_codes']} | "
        + " | ".join(str(r[f'windows_fold_{f}']) for f in range(config.N_FOLDS)) + f" | **{r['windows_total']}** | {r['window_dates_total']} |"
        for _, r in C.iterrows())
    present = sorted({int(f.split("-")[0].split("_")[1]) for f in shards})
    split_yaml = "\n".join(f"  - split: fold_{f}\n    path: data/fold_{f}-*.parquet" for f in present)
    tvt_yaml = ("  - split: train\n    path:\n" + "".join(f"    - data/fold_{f}-*.parquet\n" for f in range(3))
                + "  - split: validation\n    path: data/fold_3-*.parquet\n  - split: test\n    path: data/fold_4-*.parquet")

    card = f"""---
license: other
license_name: derived-from-nasa-and-usda-open-data
license_link: LICENSE.md
pretty_name: AVIRIS-Classic California hyperspectral time series, multilabel land cover (CDL)
task_categories:
- image-classification
task_ids:
- multi-label-image-classification
tags:
- hyperspectral
- remote-sensing
- aviris
- time-series
- earth-observation
- land-cover
- multilabel
- california
size_categories:
- 1K<n<10K
configs:
- config_name: default
  default: true
  data_files:
{split_yaml}
- config_name: train_val_test
  data_files:
{tvt_yaml}
- config_name: windows
  data_files:
  - split: train
    path: metadata/windows.parquet
- config_name: dates
  data_files:
  - split: train
    path: metadata/dates.parquet
- config_name: cdl_by_year
  data_files:
  - split: train
    path: metadata/cdl_by_year.parquet
---

# AVIRIS-C California hyperspectral time series, multilabel land cover

Machine-learning-ready **hyperspectral time series** built from NASA/JPL AVIRIS-Classic L2
surface reflectance over California (flight lines {span['min'][:4]}-{span['max'][:4]}). One sample is a fixed
**600 m x 600 m window** (32 x 32 pixels of 18.75 m) observed at **{T} dates**, with **{n_bands} cleaned
spectral bands** (from 224), a per-date quality mask, the acquisition date and flight line of
each date, and **multilabel land-cover annotations** derived from the USDA NASS Cropland Data
Layer (CDL): 12 fine classes plus `other`, given **per date** (CDL of the date's year) and **per
window**. Raw CDL proportions are shipped as well. Spatial folds of 10 km blocks support
leakage-safe cross-validation.

| | |
|---|---|
| Samples | {N} windows x {T} dates = {N*T} window-dates ({n_acq} acquired, {n_ok} pass QC) |
| Cube per sample | `(T, B, H, W) = ({T}, {n_bands}, 32, 32)`, float16, reflectance 0-1, NaN = no data |
| Spectral range | {wl[B.kept>0].min():.0f}-{wl[B.kept>0].max():.0f} nm, ~10 nm sampling (see `bands.csv`) |
| Labels | 13 classes, multilabel (presence = CDL proportion >= {config.PRESENCE_THRESHOLD:.0%}), per date and per window |
| Splits | 5 spatial folds (10 km blocks), {', '.join(f'fold_{f}={n}' for f, n in folds.items())} windows |
| Size | {size_gb:.1f} GB in {len(shards)} Parquet shards (zstd) |
| Region | California, USA ({', '.join(f'{z}: {n}' for z, n in zones.items())}) |

## 1. Structure

```
.
├── README.md                 # this card
├── bands.csv                 # 224 AVIRIS-C bands: wavelength, FWHM, kept flag, spike statistics
├── classes.csv               # 13 classes: CDL codes, window counts per fold
├── data/
│   └── fold_<k>-<i>-of-<n>.parquet   # one row = one window (cube + labels + per-date metadata)
├── metadata/
│   ├── windows.parquet|csv   # one row per window: location, zone, 10 km block, fold, window labels/proportions
│   ├── dates.parquet|csv     # one row per (window, date): date, flight line, QC flags, per-date labels/proportions
│   ├── cdl_by_year.parquet   # CDL counts of the 40 thematic groups + 13-class proportions per window and year 2008-2021
│   ├── cdl_groups.csv        # the 40 CDL groups, their CDL codes and names, and their class
│   ├── flight_lines.csv      # flight lines used (format, rotation, pixel size, ORNL file URL)
│   └── checks.json           # build checks (label statistics, fold balance, leakage, value ranges, NaN, spikes)
└── scripts/
    ├── load_example.py       # loading with `datasets`, masking, building a PyTorch-ready array
    └── sanity_check.py       # shapes, label counts, Ledoit-Wolf covariance per date (SPD test)
```

### Columns of the main configuration (`default`, one row per window)

| column | type | description |
|---|---|---|
| `window_id` | int64 | cell id `gy * 20000 + gx` on the 600 m grid aligned with the CDL 30 m grid (EPSG:5070) |
| `fold` | int8 | spatial fold 0-4 (10 km blocks) |
| `zone` | string | named box: hold-out boxes `Z2_SantaBarbara` (34.0-34.8 N, 120.5-119.2 W) and `Z4_SFBay_JasperRidge` (37.0-37.8 N, 122.6-121.8 W), study boxes `Z1_Mojave`, `Z3_Irvine`, `Z5_Sierra_Merced`, else `none` |
| `block10km` | int64 | 10.2 km block id (a block is never split across folds) |
| `lat`, `lon` | float32 | window centre (WGS84) |
| `x_min`, `y_max` | float64 | upper-left corner in EPSG:5070 (window = 600 m, pixel = 18.75 m) |
| `cube` | float16 `({T}, {n_bands}, 32, 32)` | reflectance (0-1); NaN = no data; padded dates are all-NaN |
| `dates` | 14 x string | acquisition dates `YYYY-MM-DD`, sorted; `""` = padding (window with < 14 dates) |
| `acquired` | 14 x bool | date present |
| `qc_ok` | 14 x bool | date passes QC (see below) |
| `flight_ids` | 14 x string | AVIRIS-C flight line id (`fYYMMDDtNNpNNrNN`) |
| `nan_frac` | 14 x float32 | fraction of no-data pixels |
| `n_ok_dates` | int8 | number of QC-ok dates |
| `labels_date` | int8 `({T}, {K})` | per-date multilabel: 1 present, 0 absent, -1 padded date |
| `props_date` | float32 `({T}, {K})` | per-date CDL proportions (CDL year = acquisition year); NaN for padding |
| `labels_window` | 13 x int8 | window-level multilabel (see Labels) |
| `props_window` | 13 x float32 | window-level proportions |

Class order everywhere: {', '.join(f'`{c}`' for c in config.CLASS_NAMES)}.

## 2. Acquisition and preprocessing

**Imagery.** AVIRIS-Classic L2 "calibrated reflectance" flight lines (NASA CMR collection
`{config.COLLECTION_ID}`, ORNL DAAC), {len(pd.read_csv(os.path.join(R, 'metadata', 'flight_lines.csv')))} flight lines, dates {span['min']} to {span['max']}
(acquisitions per year: {', '.join(f'{y}: {n}' for y, n in yrs.items())}). Only the pixel rows
intersecting each window were read, with HTTP Range requests.

1. *Window grid.* 600 m cells of a grid aligned on the native 30 m CDL grid (CONUS Albers,
   EPSG:5070), so that each window covers exactly 20 x 20 CDL pixels.
2. *Flight line choice.* For each date, the flight line whose CMR footprint covers the largest
   part of the window (>= {config.MIN_FLIGHT_COVERAGE:.0%}).
3. *Georeferencing.* The (sometimes 20-44 degree rotated) geotransform is read by GDAL from the
   ENVI header; the native pixels (14.8-17 m) are area-averaged onto the 32 x 32 window grid
   (`rasterio.warp.reproject`, `average`), no-data aware.
4. *Scaling.* Integer products (2013-2015, reflectance x 10000) are divided by 10000; float
   products (2016-2020) are already 0-1. Values are stored as float16.
5. *Dates.* Up to 14 dates per window, spread over the available acquisitions (farthest-point
   selection in time); failed reads were replaced by other dates when possible; windows with fewer
   dates are NaN-padded ({int((W.n_dates < T).sum())} windows).
6. *Bands.* {n_bands} of 224 bands are kept (see below).

**Band cleaning.** The base mask removes water-vapour absorption and detector-edge regions
({', '.join(base_removed)} nm, 166 bands kept). A per-band spike statistic was then measured on the
{sel['n_pairs']} QC-ok window-dates: for the window-mean spectrum, the isolated-band deviation
(peak or dip relative to both neighbours, within contiguous kept runs only), minus its temporal
median over the window's dates (removing surface features such as the red edge), counts as a spike
when larger than {config.SPIKE_REL:.0%} of the band reflectance (floor {config.SPIKE_FLOOR}). Bands whose spike
rate exceeds {config.SPIKE_RATE_FACTOR:.0f} x the median rate ({sel['median_rate_base']:.5f}) are removed:
**{', '.join(extra)} nm** ({len(extra)} bands), leaving {n_bands} bands. `bands.csv` gives, for all 224
bands, the median wavelength and FWHM over the flight-line headers, their min/max (wavelengths
differ by at most {(B.wavelength_max_nm - B.wavelength_min_nm).max():.1f} nm between flights; bands are *not* resampled),
`kept_base`, `kept`, `release_index` (position in `cube`, -1 if dropped), spike rates before/after
cleaning, the NaN rate and the share of out-of-range values.

**Quality control (per date, heuristic, not validated).** A date is `qc_ok` unless: more than
{config.QC_MAX_NAN_FRAC:.0%} no-data pixels ({qc_counts['nan_high']} window-dates), probable snow (NDSI(560, 1650 nm) > {config.QC_SNOW_NDSI} and blue > {config.QC_SNOW_BLUE},
{qc_counts['snow']}), very bright scene (blue > {config.QC_BRIGHT_BLUE} and NIR > {config.QC_BRIGHT_NIR}, {qc_counts['bright']}), or more
than half of the base-mask bands with mean <= {config.QC_ZERO_LEVEL} ({qc_counts['zero']}). Flags are in `metadata/dates.*`.

**Ground truth.** USDA NASS Cropland Data Layer 2008-2021 (30 m), read from Microsoft Planetary
Computer (collection `{config.CDL_COLLECTION}`). CDL codes are grouped into 40 thematic groups
(`metadata/cdl_groups.csv`), then into 12 fine classes + `other`. Codes outside the groups
(background, clouds/no data, non-ag/undefined) are excluded from the denominator; every
(window, year) used has at least {config.CDL_MIN_MAPPED} of 400 mapped pixels.

## 3. Labels

* **Per date** (`labels_date`): class present if its proportion in the CDL of the
  acquisition year is >= {config.PRESENCE_THRESHOLD:.0%}. Dates failing QC keep their label (QC concerns the image only).
* **Per window** (`labels_window`): proportions averaged over the *distinct* CDL years of the
  window's acquisition dates (each year counted once), thresholded at {config.PRESENCE_THRESHOLD:.0%}. This definition is
  a design choice; alternatives can be rebuilt from `props_date` or `metadata/cdl_by_year.parquet`
  (e.g. union or majority of the per-date labels).
* Labels per window: {', '.join(f'{k}: {v}' for k, v in nlab.items())} (number of classes: windows).

| id | name | description | CDL codes | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | windows | window-dates |
|---|---|---|---|---|---|---|---|---|---|---|
{cls_rows}

Counts are windows where the class is present at window level, and window-dates (all acquired
dates) where it is present at date level.

## 4. Splits

`block10km` groups the 600 m windows into 10.2 km x 10.2 km blocks; whole blocks are assigned
to 5 folds (greedy, balancing every class and the number of windows). No block and no window
is in two folds. Configurations:

* `default`: splits `fold_0` ... `fold_4` (5-fold spatial cross-validation, recommended);
* `train_val_test`: folds 0-2 / 3 / 4;
* `windows`, `dates`, `cdl_by_year`: metadata tables only (no cubes, single split `train` holding all rows), fast to load.

For a geographic-transfer test, hold out `zone == "Z2_SantaBarbara"` or `"Z4_SFBay_JasperRidge"`
and train on windows far away (e.g. > 25 km).

## 5. Loading

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("<user>/<dataset>", split="fold_0").with_format("numpy")   # or a local path
s = ds[0]
cube = s["cube"]            # (14, {n_bands}, 32, 32) reflectance, NaN = no data
ok = s["qc_ok"]             # (14,) bool
y_win = s["labels_window"]  # (13,) multilabel
y_t = s["labels_date"]      # (14, 13), -1 for padded dates
x = cube[ok]                # valid dates only

meta = load_dataset("<user>/<dataset>", "windows", split="train").to_pandas()   # no cubes
```

Streaming (`streaming=True`) avoids downloading all shards. Each cube is ~{T * n_bands * 1024 * 2 / 1e6:.1f} MB as float16;
`with_format("numpy")` returns float32 arrays (~{T * n_bands * 1024 * 4 / 1e6:.0f} MB per sample in memory). `scripts/load_example.py` and `scripts/sanity_check.py` give complete examples
(the latter computes Ledoit-Wolf spectral covariances per date and checks that they are SPD).

## 6. Known issues

* Labels come from the CDL, a 30 m classifier product, not from field data; subtypes of
  developed land and natural vegetation (shrubland, grassland, forests) are noisy, and a 30 m map
  describes an 18.75 m pixel only approximately. The 10 % threshold makes labels sensitive to small
  changes: in {int(json.load(open(os.path.join(R, 'metadata', 'checks.json')))['labels']['windows_with_date_label_changes']) if os.path.exists(os.path.join(R, 'metadata', 'checks.json')) else 'many'} windows at least one class
  flips between dates (mostly near-threshold proportions, partly real change).
* QC thresholds are heuristic. Some QC-ok dates still contain out-of-range values in kept bands
  ({chk_vals.get('window_dates_with_value_outside_m0_2_2', '?')} window-dates with a value outside [-0.2, 2]; isolated values up to 10^3-10^4, mostly
  in the SWIR 2100-2400 nm bands, on a few 2014 and 2018 flight lines): see `outlier_frac` in
  `metadata/dates.*` and `frac_outside_m0.2_2_qc_ok` in `bands.csv`. Clip (e.g. to [-0.05, 1.5])
  or mask such pixels before computing statistics. Slightly negative values
  ({chk_vals.get('frac_lt_0', float('nan')):.1%} of the values of kept bands on QC-ok dates) are kept as delivered.
* Residual date-specific artefacts remain in kept bands. With the base-mask neighbours, the
  highest spike rate of a kept band is {chk_spk.get('max_rate_final', float('nan')):.4f} (threshold {sel['threshold']:.4f}); recomputed on the final band set,
  the bands next to removed ones rise ({', '.join(f'{w:.0f} nm: {r:.3f}' for w, r in chk_spk.get('worst_final_recomputed', [])[:3])}): the 940 nm water-vapour
  feature is only partly removed (937 and 947 nm) and 928 / 957 nm still carry atmospheric
  residuals on some dates. Moderate spikes remain near 2350-2370 nm (mostly the 2020-10-13 lines).
  The band list depends on the rate factor: {'; '.join(f"x{k}: {v['n_kept']} bands" for k, v in chk_spk.get('sensitivity_factor', {}).items())}.
* Wavelength calibration differs slightly between flight lines (<= {(B.wavelength_max_nm - B.wavelength_min_nm).max():.1f} nm); bands are not resampled.
* Irregular time series: dates are concentrated in 2013-2015, gaps of several years; give the
  dates to the model. Acquisition geometry, season, atmosphere and native resolution vary.
* Spatial autocorrelation: windows of neighbouring blocks in different folds can be adjacent
  ({leak.get('frac_le_600m', float('nan')):.1%} of windows have a window of another fold at 600 m, {leak.get('frac_le_1km', float('nan')):.1%} within 1 km; median distance to the
  nearest window of another fold {leak.get('median', float('nan')) / 1000:.1f} km). 10 km folds therefore overestimate geographic
  generalisation (hold-out zones give lower scores in the source benchmark).
* All windows are in California; transfer elsewhere is untested.
* Spectral covariance of one date is ill-conditioned (n = 1024 pixels for p = {n_bands} correlated bands):
  use shrinkage (e.g. Ledoit-Wolf).

## 7. Licence

Derived from public-domain / open data: AVIRIS-Classic L2 reflectance (NASA/JPL, distributed by
the ORNL DAAC, open with citation) and the USDA NASS Cropland Data Layer (public domain). The
licence of this derived dataset is to be confirmed by the authors (see `LICENSE.md`). Please cite
the original data providers (below).

## 8. Citation

```bibtex
@misc{{aviris_ca_multilabel,
  title  = {{AVIRIS-C California hyperspectral time series, multilabel land cover}},
  author = {{TODO}},
  year   = {{TODO}},
  howpublished = {{Hugging Face dataset}},
  note   = {{TODO: DOI / paper}}
}}
```

## 9. Acknowledgements

* AVIRIS-Classic data: NASA Jet Propulsion Laboratory, distributed by the ORNL DAAC (NASA Earthdata).
* USDA National Agricultural Statistics Service Cropland Data Layer, accessed through the
  Microsoft Planetary Computer.
* Build code: see the accompanying repository (`aviris-ml build`), which rebuilds this release
  from the original archives.
"""
    with open(os.path.join(R, "README.md"), "w") as f:
        f.write(card)
    lic = """# Licence (to be confirmed)

This dataset is derived from:

* AVIRIS-Classic L2 calibrated reflectance (NASA/JPL), distributed by the ORNL DAAC. NASA
  Earth science data are openly shared without restriction; please cite the data source.
* USDA NASS Cropland Data Layer, public domain; please cite USDA NASS.

The licence under which the derived dataset itself is released has not been decided yet.
"""
    with open(os.path.join(R, "LICENSE.md"), "w") as f:
        f.write(lic)
