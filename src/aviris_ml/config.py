"""Static configuration: grids, zones, band ranges, CDL code -> class mapping, QC thresholds.

Everything that defines the dataset lives here so that a rebuild is fully determined by
this file plus the window list shipped in ``aviris_ml/data/windows.csv``.
"""
from __future__ import annotations

from importlib import resources

# ---------------------------------------------------------------------------- grid
# Windows are 600 m x 600 m cells of a grid aligned on the native 30 m CDL grid
# (CONUS Albers, EPSG:5070). Cell (gy, gx) has id gy * 20000 + gx and covers
# x in [X0 + 600 gx, X0 + 600 (gx + 1)], y in [Y0 - 600 (gy + 1), Y0 - 600 gy].
GRID_CRS = "EPSG:5070"
X0, Y0 = -2356095.0, 3172605.0
CELL_M = 600.0
CDL_RES_M = 30.0
CDL_PIX_PER_CELL = int(CELL_M / CDL_RES_M)  # 20 x 20 CDL pixels per window
NPIX = 32  # 32 x 32 AVIRIS pixels per window (18.75 m)
ID_MULT = 20000
N_DATES = 14
BLOCK_CELLS = 17  # 17 x 600 m = 10.2 km blocks for spatial folds
N_FOLDS = 5


def cell_bounds(window_id: int) -> tuple[float, float, float, float]:
    """Return (x_min, y_min, x_max, y_max) in EPSG:5070 of a window id."""
    gy, gx = divmod(int(window_id), ID_MULT)
    x_min = X0 + gx * CELL_M
    y_max = Y0 - gy * CELL_M
    return x_min, y_max - CELL_M, x_min + CELL_M, y_max


def block_id(window_id: int) -> int:
    """10 km block id (same convention as the original build: gx//17 * 100000 + gy//17)."""
    gy, gx = divmod(int(window_id), ID_MULT)
    return (gx // BLOCK_CELLS) * 100000 + gy // BLOCK_CELLS


# ---------------------------------------------------------------------------- zones
# Named study / hold-out boxes (lat_min, lat_max, lon_min, lon_max). Z2 and Z4 are the
# hold-out boxes used for the geographic-transfer benchmark of the source dataset.
ZONES = {
    "Z1_Mojave": (35.35, 35.65, -115.8, -115.3),
    "Z2_SantaBarbara": (34.0, 34.8, -120.5, -119.2),
    "Z3_Irvine": (33.5, 33.9, -118.0, -117.5),
    "Z4_SFBay_JasperRidge": (37.0, 37.8, -122.6, -121.8),
    "Z5_Sierra_Merced": (37.0, 37.4, -120.1, -119.0),
}


def zone_of(lat: float, lon: float) -> str:
    for name, (a, b, c, d) in ZONES.items():
        if a <= lat <= b and c <= lon <= d:
            return name
    return "none"


# ---------------------------------------------------------------------------- AVIRIS source
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
COLLECTION_ID = "C2711871294-ORNL_CLOUD"  # AVIRIS-Classic L2 calibrated reflectance (ORNL DAAC)
DATA_HOST_PREFIX = "https://data.ornldaac.earthdata.nasa.gov"
TEMPORAL = ("2008-01-01T00:00:00Z", "2021-12-31T23:59:59Z")
MIN_FLIGHT_COVERAGE = 0.85  # fraction of the window inside the CMR footprint
N_BANDS_RAW = 224
CLIP_ABS = 60000.0  # safety clip applied by the original extraction (absorption bands only)

# ---------------------------------------------------------------------------- bands
# Base mask (water-vapour absorption and detector edges), wavelengths in nm.
# Applied to the band *centre* wavelengths of the reference table; keeps 166 of 224 bands.
BASE_REMOVED_RANGES_NM = [(360.0, 420.0), (1300.0, 1480.0), (1780.0, 2030.0), (2440.0, 2600.0)]
# Spike statistic (see bands.py): isolated-band deviation of the window-mean spectrum, taken
# relative to the window's own temporal median, counted as a spike when larger than
# SPIKE_REL x max(reflectance, SPIKE_FLOOR). A base-mask band is dropped when its spike rate
# exceeds SPIKE_RATE_FACTOR x the median spike rate of the base-mask bands (single pass).
SPIKE_REL = 0.10
SPIKE_FLOOR = 0.05
SPIKE_REL_GROSS = 0.25  # reported only (large spikes)
SPIKE_RATE_FACTOR = 20.0
SPIKE_RATE_MIN = 0.001  # absolute floor of the threshold
# Explicit removal applied after the spike criterion: the whole 940 nm water-vapour feature
# (the spike criterion removes 937/947 nm; their wings 928/957 nm keep atmospheric residuals).
EXPLICIT_REMOVED_RANGES_NM = [(920.0, 965.0)]
# Pixel values outside this range in the base-mask bands are counted as outliers (QC column).
OUTLIER_RANGE = (-0.2, 2.0)

# ---------------------------------------------------------------------------- QC (per date)
QC_MAX_NAN_FRAC = 0.20
QC_SNOW_NDSI = 0.4
QC_SNOW_BLUE = 0.15
QC_BRIGHT_BLUE = 0.30
QC_BRIGHT_NIR = 0.40
QC_ZERO_LEVEL = 0.0005
QC_ZERO_FRAC = 0.5
QC_BANDS_NM = {"blue": 480.0, "green": 560.0, "nir": 850.0, "swir": 1650.0}

# ---------------------------------------------------------------------------- CDL ground truth
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/usda-cdl"
CDL_COLLECTION = "usda-cdl"
CDL_YEARS = list(range(2008, 2022))
CDL_MIN_MAPPED = 320  # a (window, year) is valid if >= 320 of the 400 CDL pixels map to a group

# 40 thematic groups (CDL codes not listed here, e.g. 0, 81 clouds, 88 nonag/undefined,
# are ignored: they count neither in the numerator nor in the denominator of proportions).
CDL_GROUPS: dict[str, list[int]] = {
    "eau": [83, 111, 92], "urbain_ouvert": [82, 121], "urbain_faible": [122], "urbain_moyen": [123],
    "urbain_dense": [124], "sol_nu": [65, 131], "foret_feuillue": [141, 63], "foret_coniferes": [142],
    "foret_mixte": [143], "arbustes": [64, 152], "prairie_paturage": [176, 58, 59, 60],
    "zone_humide_boisee": [190, 87], "zone_humide_herbacee": [195],
    "amandiers": [75], "noyers": [76], "pistachiers": [204], "vignes": [69], "agrumes": [72, 212],
    "oliviers": [211], "pommiers_poiriers": [68, 77], "cerisiers": [66], "pecher_nectarine": [67, 218],
    "pruniers_abricotiers": [220, 210, 223], "autres_vergers": [70, 71, 74, 215, 217, 242, 250, 55],
    "mais": [1, 12, 13], "ble_orge_avoine": [21, 22, 23, 24, 25, 27, 28, 29, 30, 205, 224], "coton": [2],
    "riz": [3], "sorgho_soja_tournesol": [4, 5, 6, 10],
    "oleagineux_legumineuses": [31, 32, 33, 34, 35, 38, 39, 42, 51, 52, 53],
    "luzerne": [36], "foin_autre": [37], "tomates": [54], "oignons_ail": [49, 208],
    "pommes_de_terre_betteraves": [43, 41, 46], "salades_legumes_feuilles": [227, 219, 243, 244, 245, 246, 247],
    "autres_legumes": [44, 45, 47, 48, 50, 57, 206, 207, 209, 213, 214, 216, 221, 222, 229, 248, 249, 11, 14, 56],
    "jachere": [61],
    "double_culture": [26, 225, 226, 228, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 254],
    "neige_glace": [112],
}
GROUP_NAMES = list(CDL_GROUPS)
# English names of the 40 groups (used in the release files; internal ids are kept in config)
GROUP_EN = {
    "eau": "water", "urbain_ouvert": "developed_open", "urbain_faible": "developed_low", "urbain_moyen": "developed_medium",
    "urbain_dense": "developed_high", "sol_nu": "barren", "foret_feuillue": "deciduous_forest", "foret_coniferes": "evergreen_forest",
    "foret_mixte": "mixed_forest", "arbustes": "shrubland", "prairie_paturage": "grassland_pasture", "zone_humide_boisee": "woody_wetlands",
    "zone_humide_herbacee": "herbaceous_wetlands", "amandiers": "almond", "noyers": "walnut", "pistachiers": "pistachio", "vignes": "grapes",
    "agrumes": "citrus", "oliviers": "olive", "pommiers_poiriers": "apple_pear", "cerisiers": "cherry", "pecher_nectarine": "peach_nectarine",
    "pruniers_abricotiers": "plum_apricot", "autres_vergers": "other_orchards", "mais": "corn", "ble_orge_avoine": "small_grains",
    "coton": "cotton", "riz": "rice", "sorgho_soja_tournesol": "sorghum_soy_sunflower", "oleagineux_legumineuses": "oilseeds_legumes",
    "luzerne": "alfalfa", "foin_autre": "other_hay", "tomates": "tomatoes", "oignons_ail": "onion_garlic",
    "pommes_de_terre_betteraves": "potato_sugarbeet", "salades_legumes_feuilles": "lettuce_leafy_vegetables",
    "autres_legumes": "other_vegetables", "jachere": "fallow", "double_culture": "double_crop", "neige_glace": "snow_ice",
}

# 12 fine classes + "other" (all remaining groups). Internal group id -> English class name.
CLASSES = [
    ("amandiers", "almond"), ("arbustes", "shrubland"), ("eau", "water"),
    ("foret_coniferes", "evergreen_forest"), ("foret_mixte", "mixed_forest"),
    ("prairie_paturage", "grassland_pasture"), ("sol_nu", "barren"),
    ("urbain_dense", "developed_high"), ("urbain_faible", "developed_low"),
    ("urbain_moyen", "developed_medium"), ("urbain_ouvert", "developed_open"),
    ("vignes", "grapes"),
]
CLASS_NAMES = [en for _, en in CLASSES] + ["other"]
N_CLASSES = len(CLASS_NAMES)
PRESENCE_THRESHOLD = 0.10


def group_to_class_index() -> list[int]:
    """For each of the 40 groups, the index of its class among the 13 output classes."""
    fine = [g for g, _ in CLASSES]
    return [fine.index(g) if g in fine else N_CLASSES - 1 for g in GROUP_NAMES]


def class_cdl_codes() -> list[list[int]]:
    """CDL codes of each of the 13 classes."""
    out: list[list[int]] = [[] for _ in range(N_CLASSES)]
    for g, ci in zip(GROUP_NAMES, group_to_class_index()):
        out[ci].extend(CDL_GROUPS[g])
    return [sorted(c) for c in out]


def cdl_lut():
    """uint8 lookup table CDL code -> group index (255 = ignored)."""
    import numpy as np

    lut = np.full(256, 255, np.uint8)
    for i, g in enumerate(GROUP_NAMES):
        for c in CDL_GROUPS[g]:
            lut[c] = i
    return lut


def packaged_windows_csv() -> str:
    """Path of the shipped window list (2034 windows, 14 dates each, plus all candidate dates)."""
    return str(resources.files("aviris_ml") / "data" / "windows.csv")


def packaged_cdl_legend() -> str:
    return str(resources.files("aviris_ml") / "data" / "cdl_legend.json")


def packaged(name: str) -> str:
    """Packaged reference files: bands_reference.csv (band table from the 975 flight-line
    headers of the original build), flight_lines_reference.csv (header info + ORNL URL of the
    flight lines used), cdl_counts_reference.npz (CDL group counts 2008-2021 of the 2034 windows)."""
    return str(resources.files("aviris_ml") / "data" / name)
