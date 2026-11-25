# aviris-ca-multilabel

Build code for the **AVIRIS-C California hyperspectral time-series multilabel dataset**:
2034 windows of 600 m x 600 m (32 x 32 pixels), 14 dates each, AVIRIS-Classic L2 reflectance
with cleaned bands, per-date QC, and multilabel land-cover labels (12 classes + `other`) from the
USDA Cropland Data Layer, exported as Hugging Face-ready Parquet shards with a dataset card.

The pipeline rebuilds the release end to end from the public archives:

| stage | module | what it does | output (work dir) |
|---|---|---|---|
| `download` | `cmr`, `download` | CMR search of AVIRIS-C L2 flight lines (`C2711871294-ORNL_CLOUD`), choice of the flight line covering each window, ENVI header, **one HTTP Range request per (window, date)** for the image rows crossing the window; only the columns around the window are kept | `cmr_flights.json`, `hdr/`, `raw/<window>/<date>.npy+json` |
| `assemble` | `assemble` | geotransform (incl. 20-44 degree rotations) from the header through GDAL, scaling to 0-1, no-data mask, area-average reprojection onto the 32 x 32 EPSG:5070 window grid, float16 | `cubes/<window>.npz`, `cubes_index.json` |
| `qc` | `qc`, `bands` | per-date QC flags, per-date spectral statistics, band table (wavelength/FWHM from the headers), base absorption mask (166 bands) and spike screening (final mask) | `qc_stats.npz`, `bands.csv`, `bands_selection.json` |
| `groundtruth` | `groundtruth` | CDL 2008-2021 from Planetary Computer (`usda-cdl`): exact 20 x 20 block of 30 m pixels per window and year, counts of 40 thematic groups | `cdl_counts.npz` |
| `labels` | `labels`, `splits` | yearly / per-date / window proportions of the 13 classes, multilabels (>= 10 %), spatial 10 km folds | `labels.npz` |
| `export` | `export`, `checks`, `card` | Parquet shards per fold, metadata tables, `bands.csv`, `classes.csv`, checks, README card, scripts | release directory |

Every stage is idempotent (atomic writes; existing outputs are reused, `--force` recomputes)
and resumable (downloads are skipped when the file exists with the expected size).

## Install

```bash
uv sync --extra dev          # Python >= 3.11; GDAL comes with the rasterio wheel
uv run pytest -q             # tests on tiny synthetic data (no network)
```

## Credentials

* **Earthdata Login** (ORNL DAAC): put a `machine urs.earthdata.nasa.gov login ... password ...`
  entry in `~/.netrc` (mode 600). `requests` reads it when ORNL redirects to URS; the code never
  reads, prints, logs or copies the credentials, and session cookies stay in memory.
* **Planetary Computer**: anonymous SAS token fetched at run time; signed URLs contain the token
  and are never logged (HTTP client loggers are set to WARNING).

## Rebuild

### From the already extracted cubes (what was used for the current release)

```bash
uv run aviris-ml build --stage all \
    --from-existing /DATA/DATASET/HYPERSPECTRAL/AVIRIS-C_CA_TS14_MIX \
    --work ~/projets/aviris-ca-multilabel-work --release ~/projets/aviris-ca-multilabel-release --workers 3
uv run aviris-ml check --from-existing /DATA/DATASET/HYPERSPECTRAL/AVIRIS-C_CA_TS14_MIX   # re-run checks + card
```

`--from-existing` only reads the source folder (cubes, and `meta/` for consistency checks).
Small reference files of the original build are packaged in `src/aviris_ml/data/` and used by
default, so this command needs nothing else:

* `bands_reference.csv`: median wavelength / FWHM over the 975 flight-line headers (overridden
  by `--headers DIR` or by the headers fetched by the `download` stage);
* `cdl_counts_reference.npz`: CDL counts of the 40 groups, 2008-2021, for the 2034 windows
  (overridden by `--cdl-cache`; in network mode the `groundtruth` stage fetches the CDL);
  rebuilt from the original cache with `tools/extract_cdl_cache.py`;
* `flight_lines_reference.csv`: format, rotation, pixel size and ORNL URL of the 943 flight lines.

A rebuild with this command into a fresh work directory gives a release identical to the
published one (all Parquet shards and tables byte-identical, see `RAPPORT_BUILD.md`).

### From the archives (network)

```bash
uv run aviris-ml build --stage download --stage assemble --dl-workers 2 --work WORK
uv run aviris-ml build --stage qc --stage groundtruth --stage labels --stage export --work WORK --release RELEASE
```

The window list and the dates of each window are shipped in `src/aviris_ml/data/windows.csv`
(`dates` = the 14 dates of the release, `dates_all` = all candidate dates, used as replacements
when a read fails). `--window-ids`, `--limit` and `--max-dates` restrict a run for tests.

Proof run on 2 windows (5 dates: uint16 BIP 2013, int16 BIP 2014, float32 BIL 2016/2018,
rotated lines), then comparison with the source cubes and CDL cache:

```bash
uv run aviris-ml build --stage download --stage assemble --stage groundtruth \
    --windows-csv tools/windows_proof.csv --work WORK/proof --dl-workers 2
uv run python tools/compare_with_source.py WORK/proof /DATA/DATASET/HYPERSPECTRAL/AVIRIS-C_CA_TS14_MIX \
    --cdl-ref WORK/existing/cdl_counts_mix2034.npz
```

Result: 153 MB read in 16 s; all 5 window-dates bit-identical (same flight line, same NaN
pattern, max |diff| = 0) and CDL counts identical for 2008-2021.

### Time and disk estimates

| step | full dataset (2034 windows) |
|---|---|
| download | ~0.5 GB read per window (whole image rows), **~1 TB** in total; ~6 h with 6 concurrent requests in the original build (be polite: 2-6). On disk: ~1-2 MB per date (cropped blocks, deleted after assembly unless `--keep-raw`, ~40 GB if kept) |
| assemble | cubes ~4 MB per window, ~8.5 GB |
| qc | ~75 s with 3 processes |
| groundtruth | one STAC search per window + one COG read per (year, tile); minutes to ~1 h depending on the network |
| labels | seconds |
| export | ~2 min with 3 processes; release ~7 GB |

## Decisions (validated; see the dataset card for details)

* Windows: the 2034 windows of `AVIRIS-C_CA_TS14_MIX`, 14 dates each (171 windows have padded dates).
* Bands: base mask (365-414, 1303-1473, 1782-2028, 2447-2496 nm removed: 166 bands), then
  spike screening (`bands.py`): isolated-band deviation of the window-mean spectrum minus its
  temporal median, spike if > 10 % of max(reflectance, 0.05); bands whose spike rate exceeds 20 x
  the median rate are removed (9 bands), then the 940 nm water-vapour feature is removed explicitly
  (920-965 nm: 928 and 957 nm in addition to the spike-flagged 938/947 nm). **155 bands** (validated).
* Labels: class present if >= 10 % of the mapped CDL pixels. Per date: CDL of the acquisition
  year. Window: mean of the yearly proportions over the distinct years of the window's dates.
* Splits: 10.2 km blocks (17 x 17 windows) assigned greedily to 5 folds, balancing classes.
* Out-of-range values are kept and flagged (`outlier_frac`), not masked.
* Licences: dataset CC BY 4.0 (attribution to NASA/JPL AVIRIS-Classic via ORNL DAAC and USDA NASS
  CDL, see the release `LICENSE.md`); code MIT (`LICENSE`).

## Layout

```
src/aviris_ml/   config cmr download envi assemble qc bands groundtruth labels splits export checks card pipeline cli
src/aviris_ml/data/             windows.csv (window list), cdl_legend.json
src/aviris_ml/release_scripts/  load_example.py, sanity_check.py (copied into the release)
tests/                          pytest, synthetic data only
tools/compare_with_source.py    rebuild-vs-source comparison
```
