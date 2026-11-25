"""Minimal ENVI header parsing and georeferencing through a sparse stub file."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

DTYPES = {2: "<i2", 12: "<u2", 4: "<f4"}


@dataclass
class EnviHeader:
    samples: int
    lines: int
    bands: int
    dtype: int
    interleave: str
    byte_order: int
    offset: int
    wavelength: np.ndarray | None
    fwhm: np.ndarray | None
    rotation: float
    pixel_size: float
    text: str

    @property
    def bytes_per_sample(self) -> int:
        return np.dtype(DTYPES[self.dtype]).itemsize

    @property
    def row_bytes(self) -> int:
        return self.samples * self.bands * self.bytes_per_sample

    @property
    def total_bytes(self) -> int:
        return self.row_bytes * self.lines

    @property
    def supported(self) -> bool:
        return self.interleave in ("bip", "bil") and self.dtype in DTYPES and self.byte_order == 0 and self.offset == 0

    @property
    def scale(self) -> float:
        """Reflectance scale: integer products are reflectance x 10000, float products are 0-1."""
        return 1.0 if self.dtype == 4 else 1e-4


def _key(text: str, name: str, cast=str):
    m = re.search(rf"^{name}\s*=\s*(.+)$", text, re.M | re.I)
    return cast(m.group(1).strip()) if m else None


def _vector(text: str, name: str) -> np.ndarray | None:
    m = re.search(rf"^{name}\s*=\s*\{{([^}}]*)\}}", text, re.S | re.M | re.I)
    if not m:
        return None
    return np.array([float(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()])


def parse_header(text: str) -> EnviHeader:
    rot = 0.0
    px = float("nan")
    mi = re.search(r"^map info\s*=\s*\{([^}]*)\}", text, re.M | re.I)
    if mi:
        parts = [p.strip() for p in mi.group(1).split(",")]
        try:
            px = float(parts[5])
        except (IndexError, ValueError):
            pass
        r = re.search(r"rotation\s*=\s*([-\d.]+)", mi.group(1))
        rot = float(r.group(1)) if r else 0.0
    return EnviHeader(
        samples=_key(text, "samples", int), lines=_key(text, "lines", int), bands=_key(text, "bands", int),
        dtype=_key(text, "data type", int), interleave=(_key(text, "interleave") or "").lower(),
        byte_order=_key(text, "byte order", int) or 0, offset=_key(text, "header offset", int) or 0,
        wavelength=_vector(text, "wavelength"), fwhm=_vector(text, "fwhm"), rotation=rot, pixel_size=px, text=text,
    )


def georef_from_stub(hdr: EnviHeader, stub_dir: str, name: str):
    """Write a sparse stub ``<name>.bin`` + header and let GDAL derive the (possibly rotated)
    geotransform from the ENVI 'map info'. Returns (affine transform, CRS)."""
    import rasterio

    os.makedirs(stub_dir, exist_ok=True)
    stub = os.path.join(stub_dir, name + ".bin")
    if not os.path.exists(stub) or os.path.getsize(stub) != hdr.total_bytes:
        with open(stub, "wb") as f:
            f.truncate(hdr.total_bytes)  # sparse: no disk usage
        with open(os.path.join(stub_dir, name + ".hdr"), "w") as f:
            f.write(hdr.text)
    with rasterio.open(stub) as ds:
        return ds.transform, ds.crs


def decode_rows(raw: bytes | np.ndarray, hdr: EnviHeader, nrows: int) -> np.ndarray:
    """Decode a block of full image rows into (rows, samples, bands) native dtype."""
    a = np.frombuffer(raw, DTYPES[hdr.dtype]) if isinstance(raw, (bytes, bytearray)) else raw
    if hdr.interleave == "bip":
        return a.reshape(nrows, hdr.samples, hdr.bands)
    return a.reshape(nrows, hdr.bands, hdr.samples).swapaxes(1, 2)
