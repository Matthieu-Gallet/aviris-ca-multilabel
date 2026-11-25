"""Flight-line search in NASA CMR (AVIRIS-Classic L2 reflectance, collection C2711871294-ORNL_CLOUD)."""
from __future__ import annotations

import json
import logging
import os
import re
import time

import requests

from . import config

log = logging.getLogger(__name__)
FLIGHT_RE = re.compile(r"f\d{6}t\d+p\d+r\d+")


def _get(params: dict, retries: int = 5) -> dict:
    for k in range(retries):
        try:
            r = requests.get(config.CMR_URL, params=params, timeout=300)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:  # pragma: no cover - network
            log.warning("CMR request failed (%s), retry %d", type(e).__name__, k + 1)
            time.sleep(4 * (k + 1))
    raise RuntimeError("CMR search failed")


def parse_entry(g: dict) -> tuple[str, dict] | None:
    """Turn one CMR granule entry into (flight_id, record) or None if no reflectance .bin link."""
    m = FLIGHT_RE.search(g.get("title", ""))
    if not m:
        return None
    bins = sorted({l["href"] for l in g.get("links", []) if l.get("href", "").startswith(config.DATA_HOST_PREFIX)
                   and "corr" in l["href"] and l["href"].endswith(".bin")})
    if not bins:
        return None
    polys = []
    for p in g.get("polygons") or []:
        v = [float(t) for t in p[0].split()]
        polys.append([(v[i + 1], v[i]) for i in range(0, len(v) - 1, 2)])  # (lon, lat)
    return m.group(0), {"date": g["time_start"][:10], "url": bins[0], "polygons_lonlat": polys}


def search_flights(cache: str | None = None, temporal=config.TEMPORAL, bbox: str | None = None) -> dict:
    """All AVIRIS-C L2 reflectance flight lines in the time range. Cached as JSON when ``cache`` is given."""
    if cache and os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    flights: dict = {}
    page = 1
    while True:
        params = {"collection_concept_id": config.COLLECTION_ID, "page_size": 2000, "page_num": page,
                  "temporal": f"{temporal[0]},{temporal[1]}"}
        if bbox:
            params["bounding_box"] = bbox
        entries = _get(params).get("feed", {}).get("entry", [])
        if not entries:
            break
        for g in entries:
            rec = parse_entry(g)
            if rec:
                flights[rec[0]] = rec[1]
        page += 1
    log.info("CMR: %d flight lines with reflectance files", len(flights))
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        tmp = cache + ".part"
        with open(tmp, "w") as f:
            json.dump(flights, f)
        os.replace(tmp, cache)
    return flights


def flight_polygons_5070(flights: dict) -> dict:
    """Footprints projected to EPSG:5070 (shapely polygons)."""
    from pyproj import Transformer
    from shapely.geometry import Polygon

    tr = Transformer.from_crs(4326, 5070, always_xy=True)
    out = {}
    for fid, rec in flights.items():
        ps = []
        for ring in rec["polygons_lonlat"]:
            xs, ys = tr.transform([p[0] for p in ring], [p[1] for p in ring])
            ps.append(Polygon(list(zip(xs, ys))).buffer(0))
        out[fid] = ps
    return out


def best_flight(window_bounds, date: str, flights: dict, polys: dict, by_date: dict) -> tuple[str | None, float]:
    """Flight line of ``date`` covering the largest fraction of the window."""
    from shapely.geometry import box

    wb = box(*window_bounds)
    best, cov = None, 0.0
    for fid in by_date.get(date, []):
        c = sum(p.intersection(wb).area for p in polys[fid]) / wb.area
        if c > cov:
            best, cov = fid, c
    return best, cov
