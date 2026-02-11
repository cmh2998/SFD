# app.py
from fastapi.responses import HTMLResponse
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException
import requests
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os

from shapely.geometry import Point, MultiPoint, shape, mapping
from shapely.prepared import prep
from shapely.ops import transform as shp_transform
from pyproj import Transformer


app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
BOUNDARY_PATH = BASE_DIR / "county_region.GeoJSON"

# Optional, but useful on some setups
os.environ["SHAPE_RESTORE_SHX"] = "YES"

# ----------------------------
# Simple in-memory cache
# ----------------------------
_AREA_CACHE = {
    "ts": 0.0,
    "poly": None,       # shapely geometry in EPSG:4326
    "bbox": None,       # dict in EPSG:4326
    "includes": None,   # list of included names
    "prepared": None,   # prepared geometry in EPSG:4326
}
_AREA_TTL_SECONDS = 3600  # 1 hour

_STATIONS_CACHE = {"ts": 0.0, "data": None}
_STATIONS_TTL_SECONDS = 600  # 10 minutes

_MEASURES_CACHE = {}  # station_id -> {"ts": float, "items": list}
_MEASURES_TTL_SECONDS = 300  # 5 minutes

_STATION_META_CACHE = {}  # station_id -> {"ts": float, "meta": dict}
_STATION_META_TTL_SECONDS = 3600  # 1 hour

_CATCHMENTS_CACHE = {"ts": 0.0, "geojson": None}
_CATCHMENTS_TTL_SECONDS = 3600  # 1 hour

_FLOODWARN_CACHE = {"ts": 0.0, "geojson": None}
_FLOODWARN_TTL_SECONDS = 300  # 5 minutes

OFFLINE_AFTER_HOURS = 6


def _parse_dt(dt_str):
    if not dt_str:
        return None
    try:
        if dt_str.endswith("Z"):
            dt_str = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None


def _normalise_unit_to_short(unit):
    if not unit:
        return None

    raw = str(unit).strip()
    cleaned = re.sub(r"\s+", " ", raw)
    cleaned = re.sub(r"[^A-Za-z0-9/\^\- ]+", "", cleaned).strip()

    if not cleaned:
        return None

    lower = cleaned.lower()
    if lower.startswith("m") and not lower.startswith("mm"):
        return "m"

    token = cleaned.split(" ")[0]
    u = token.lower()

    if u in ["m", "metre", "metres", "meter", "meters"]:
        return "m"
    if u in ["mm", "millimetre", "millimetres", "millimeter", "millimeters"]:
        return "mm"

    return token or None


def _detect_epsg_from_geojson(gj: dict) -> int:
    """
    Tries to detect EPSG from GeoJSON "crs".
    Defaults to EPSG:27700 for UK boundary datasets like yours.
    """
    try:
        crs = gj.get("crs") or {}
        props = crs.get("properties") or {}
        name = props.get("name") or ""
        name = str(name)

        # Example: "urn:ogc:def:crs:EPSG::27700"
        m = re.search(r"EPSG::(\d+)", name)
        if m:
            return int(m.group(1))

        # Example: "EPSG:27700"
        m2 = re.search(r"EPSG:(\d+)", name)
        if m2:
            return int(m2.group(1))
    except Exception:
        pass

    return 27700


def _to_wgs84_geom(geom_dict: dict, source_epsg: int):
    """
    Convert geometry dict to shapely in EPSG:4326.
    """
    g = shape(geom_dict)

    if source_epsg == 4326:
        return g

    transformer = Transformer.from_crs(f"EPSG:{source_epsg}", "EPSG:4326", always_xy=True)

    def _proj(x, y, z=None):
        return transformer.transform(x, y)

    return shp_transform(_proj, g)


def _read_boundary_geojson():
    try:
        raw = BOUNDARY_PATH.read_text(encoding="utf-8")
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read boundary GeoJSON at {BOUNDARY_PATH}: {e}")


def _pick_name(props: dict) -> str:
    for k in ["LAD17NM", "LAD22NM", "LAD21NM", "LAD20NM", "LAD19NM", "LAD18NM", "NAME", "Name", "name"]:
        v = props.get(k)
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return ""


def _load_staffs_plus_stoke_polygon_and_bbox():
    now = time.time()
    if (
        _AREA_CACHE["poly"] is not None
        and _AREA_CACHE["bbox"] is not None
        and (now - _AREA_CACHE["ts"]) < _AREA_TTL_SECONDS
    ):
        return _AREA_CACHE["poly"], _AREA_CACHE["bbox"], _AREA_CACHE["includes"], _AREA_CACHE["prepared"]

    gj = _read_boundary_geojson()
    source_epsg = _detect_epsg_from_geojson(gj)

    features = gj.get("features") or []
    if not isinstance(features, list):
        raise HTTPException(status_code=500, detail="Boundary GeoJSON features is not a list")

    wanted = ["staffordshire", "stoke-on-trent", "stoke on trent", "stoke-on trent"]

    kept_geoms = []
    kept_names = []

    for f in features:
        if not isinstance(f, dict):
            continue
        props = f.get("properties") or {}
        nm = _pick_name(props)
        if not nm:
            continue

        low = nm.lower()
        if any(w in low for w in wanted):
            geom_dict = f.get("geometry")
            if not geom_dict:
                continue
            try:
                g4326 = _to_wgs84_geom(geom_dict, source_epsg)
                kept_geoms.append(g4326)
                kept_names.append(nm)
            except Exception:
                continue

    if not kept_geoms:
        raise HTTPException(
            status_code=404,
            detail="Could not find Staffordshire or Stoke-on-Trent in boundary GeoJSON properties",
        )

    area_poly = kept_geoms[0]
    for g in kept_geoms[1:]:
        area_poly = area_poly.union(g)

    try:
        if hasattr(area_poly, "is_valid") and not area_poly.is_valid:
            area_poly = area_poly.buffer(0)
    except Exception:
        pass

    minx, miny, maxx, maxy = area_poly.bounds
    bbox = {"minLon": float(minx), "minLat": float(miny), "maxLon": float(maxx), "maxLat": float(maxy)}
    includes = sorted(list({n for n in kept_names if n}))

    _AREA_CACHE["ts"] = now
    _AREA_CACHE["poly"] = area_poly
    _AREA_CACHE["bbox"] = bbox
    _AREA_CACHE["includes"] = includes
    _AREA_CACHE["prepared"] = prep(area_poly)

    return area_poly, bbox, includes, _AREA_CACHE["prepared"]


def _get_station_measures(station_id):
    now = time.time()
    cached = _MEASURES_CACHE.get(station_id)
    if cached and (now - cached["ts"]) < _MEASURES_TTL_SECONDS:
        return cached["items"]

    url = f"https://environment.data.gov.uk/flood-monitoring/id/stations/{station_id}/measures"
    r = requests.get(url, timeout=8)
    if r.status_code != 200:
        items = []
    else:
        data = r.json()
        items = data.get("items", []) or []

    _MEASURES_CACHE[station_id] = {"ts": now, "items": items}
    return items


def _get_station_meta(station_id):
    now = time.time()
    cached = _STATION_META_CACHE.get(station_id)
    if cached and (now - cached["ts"]) < _STATION_META_TTL_SECONDS:
        return cached["meta"]

    meta = {"riverName": None, "catchmentName": None}

    try:
        url = f"https://environment.data.gov.uk/flood-monitoring/id/stations/{station_id}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            data = r.json()
            item = data.get("items")
            if isinstance(item, dict):
                meta["riverName"] = item.get("riverName")
                meta["catchmentName"] = item.get("catchmentName")
    except Exception:
        pass

    _STATION_META_CACHE[station_id] = {"ts": now, "meta": meta}
    return meta


def _pick_latest_measure(measures):
    best = None
    best_dt = None

    for m in measures:
        if not isinstance(m, dict):
            continue
        latest = m.get("latestReading") or {}
        dt = _parse_dt(latest.get("dateTime"))
        if dt is None:
            continue
        if best_dt is None or dt > best_dt:
            best = m
            best_dt = dt

    return best, best_dt


def _state_for_measure(parameter, unit_short, value):
    try:
        v = float(value)
    except Exception:
        return "NoData"

    p = (parameter or "").lower()
    u = (unit_short or "").lower()

    if p == "level" or u == "m":
        if v > 1.1:
            return "High"
        if v > 0.8:
            return "Elevated"
        return "Normal"

    if p == "rainfall" or u == "mm":
        if v >= 10:
            return "High"
        if v >= 2:
            return "Elevated"
        return "Normal"

    if p == "flow" or "m3/s" in u or "cumecs" in u:
        if v >= 50:
            return "High"
        if v >= 20:
            return "Elevated"
        return "Normal"

    if v > 0:
        return "Normal"
    return "NoData"


def _compute_trend_for_measure(measure_id):
    try:
        since_dt = datetime.now(timezone.utc) - timedelta(hours=2)
        since = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        measure_key = str(measure_id).split("/")[-1]
        tr = requests.get(
            f"https://environment.data.gov.uk/flood-monitoring/id/measures/{measure_key}/readings?_sorted&since={since}",
            timeout=8,
        )
        if tr.status_code != 200:
            return "NoData", None, None

        tj = tr.json()
        hist = tj.get("items", [])
        if len(hist) < 2:
            return "NoData", None, None

        latest_val = hist[0].get("value")
        old_val = hist[-1].get("value")
        latest_dt = _parse_dt(hist[0].get("dateTime"))
        old_dt = _parse_dt(hist[-1].get("dateTime"))

        if latest_val is None or old_val is None or latest_dt is None or old_dt is None:
            return "NoData", None, None

        diff = float(latest_val) - float(old_val)
        hours = max(0.0001, (latest_dt - old_dt).total_seconds() / 3600.0)
        rate = diff / hours

        if diff > 0.01:
            trend = "Rising"
        elif diff < -0.01:
            trend = "Falling"
        else:
            trend = "Steady"

        return trend, round(diff, 4), round(rate, 4)
    except Exception:
        return "NoData", None, None


def _build_catchments_geojson():
    now = time.time()
    if _CATCHMENTS_CACHE["geojson"] is not None and (now - _CATCHMENTS_CACHE["ts"]) < _CATCHMENTS_TTL_SECONDS:
        return _CATCHMENTS_CACHE["geojson"]

    stations_payload = staffordshire_stations()
    stations = stations_payload.get("stations", []) or []

    by_catch = {}
    for st in stations:
        sid = st.get("id")
        lat = st.get("lat")
        lon = st.get("long")
        if not sid or lat is None or lon is None:
            continue

        meta = _get_station_meta(sid)
        cn = meta.get("catchmentName") or "Unknown"
        by_catch.setdefault(cn, []).append((float(lon), float(lat)))

    features = []
    for cn, pts in by_catch.items():
        if len(pts) < 3:
            continue
        hull = MultiPoint(pts).convex_hull
        if hull.is_empty:
            continue

        features.append(
            {
                "type": "Feature",
                "properties": {"catchmentName": cn, "stationCount": len(pts)},
                "geometry": mapping(hull),
            }
        )

    out = {"type": "FeatureCollection", "features": features}
    _CATCHMENTS_CACHE["ts"] = now
    _CATCHMENTS_CACHE["geojson"] = out
    return out


def _build_flood_warnings_geojson():
    now = time.time()
    if _FLOODWARN_CACHE["geojson"] is not None and (now - _FLOODWARN_CACHE["ts"]) < _FLOODWARN_TTL_SECONDS:
        return _FLOODWARN_CACHE["geojson"]

    _, bbox, _, _ = _load_staffs_plus_stoke_polygon_and_bbox()
    minLon = bbox["minLon"]
    minLat = bbox["minLat"]
    maxLon = bbox["maxLon"]
    maxLat = bbox["maxLat"]

    url = (
        "https://environment.data.gov.uk/flood-monitoring/id/floods"
        f"?min-lat={minLat}&max-lat={maxLat}&min-long={minLon}&max-long={maxLon}"
    )

    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            out = {"type": "FeatureCollection", "features": []}
            _FLOODWARN_CACHE["ts"] = now
            _FLOODWARN_CACHE["geojson"] = out
            return out

        data = r.json()
        items = data.get("items", []) or []
    except Exception:
        items = []

    features = []
    for it in items:
        fa = it.get("floodArea") or {}
        poly = fa.get("polygon")
        if not poly:
            continue

        geom = None
        if isinstance(poly, dict) and poly.get("type") and poly.get("coordinates"):
            geom = poly

        if geom is None:
            continue

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "severity": it.get("severity"),
                    "severityLevel": it.get("severityLevel"),
                    "message": it.get("message"),
                    "timeRaised": it.get("timeRaised"),
                    "timeMessageChanged": it.get("timeMessageChanged"),
                    "area": fa.get("label") or fa.get("fwdCode") or "Flood area",
                },
                "geometry": geom,
            }
        )

    out = {"type": "FeatureCollection", "features": features}
    _FLOODWARN_CACHE["ts"] = now
    _FLOODWARN_CACHE["geojson"] = out
    return out


def _fetch_station_row(st):
    sid = st.get("id")
    label = st.get("label")
    lat = st.get("lat")
    lon = st.get("long")

    row = {
        "id": sid,
        "label": label,
        "lat": lat,
        "long": lon,
        "district": None,
        "riverName": None,
        "catchmentName": None,
        "measureId": None,
        "parameter": None,
        "unit": None,
        "unitShort": None,
        "dateTime": None,
        "value": None,
        "height": None,
        "trend": "NoData",
        "trendDiff": None,
        "trendRatePerHour": None,
        "state": "NoData",
        "reporting": "NoData",
        "ageSeconds": None,
    }

    if not sid:
        return row

    meta = _get_station_meta(sid)
    row["riverName"] = meta.get("riverName")
    row["catchmentName"] = meta.get("catchmentName")

    measures = _get_station_measures(sid)
    best, best_dt = _pick_latest_measure(measures)

    if not best or not best_dt:
        row["reporting"] = "NoData"
        return row

    measure_id = best.get("measure") or best.get("@id")
    latest = best.get("latestReading") or {}

    value = latest.get("value")
    dt_str = latest.get("dateTime")

    unit = latest.get("unitName") or best.get("unitName")
    unit_short = _normalise_unit_to_short(unit)
    parameter = best.get("parameter")

    row["measureId"] = measure_id
    row["parameter"] = parameter
    row["unit"] = unit
    row["unitShort"] = unit_short
    row["dateTime"] = dt_str
    row["value"] = value

    if unit_short == "m":
        row["height"] = value

    now_dt = datetime.now(timezone.utc)
    age = (now_dt - best_dt).total_seconds()
    row["ageSeconds"] = int(age)

    if age > (OFFLINE_AFTER_HOURS * 3600):
        row["reporting"] = "Offline"
    else:
        row["reporting"] = "Live"

    row["state"] = _state_for_measure(parameter, unit_short, value)

    if measure_id:
        trend, diff, rate = _compute_trend_for_measure(measure_id)
        row["trend"] = trend
        row["trendDiff"] = diff
        row["trendRatePerHour"] = rate

    return row


# ----------------------------
# Existing endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/station/{station_id}")
def station_reading(station_id: str):
    url = f"https://environment.data.gov.uk/flood-monitoring/id/stations/{station_id}/readings"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    items = data.get("items", [])
    if not items:
        raise HTTPException(status_code=404, detail="No readings found for this station")

    item = items[0]
    return {"station": station_id, "dateTime": item.get("dateTime"), "value": item.get("value"), "unit": item.get("unitName")}


# ----------------------------
# Staffordshire + Stoke endpoints
# ----------------------------
@app.get("/staffordshire/bbox")
def staffordshire_bbox():
    _, bbox, includes, _ = _load_staffs_plus_stoke_polygon_and_bbox()
    return {
        "minLon": bbox["minLon"],
        "minLat": bbox["minLat"],
        "maxLon": bbox["maxLon"],
        "maxLat": bbox["maxLat"],
        "includes": includes,
    }


@app.get("/staffordshire/boundary")
def staffordshire_boundary():
    """
    Returns a WGS84 (EPSG:4326) GeoJSON so Leaflet can draw it.
    """
    gj = _read_boundary_geojson()
    source_epsg = _detect_epsg_from_geojson(gj)

    out_features = []
    for f in gj.get("features") or []:
        if not isinstance(f, dict):
            continue
        geom_dict = f.get("geometry")
        if not geom_dict:
            continue

        try:
            g4326 = _to_wgs84_geom(geom_dict, source_epsg)
            out_features.append(
                {
                    "type": "Feature",
                    "properties": f.get("properties") or {},
                    "geometry": mapping(g4326),
                }
            )
        except Exception:
            continue

    return {"type": "FeatureCollection", "features": out_features}


@app.get("/staffordshire/districts")
def staffordshire_districts():
    return {"type": "FeatureCollection", "features": []}


@app.get("/staffordshire/catchments")
def staffordshire_catchments():
    return _build_catchments_geojson()


@app.get("/staffordshire/floodwarnings")
def staffordshire_floodwarnings():
    return _build_flood_warnings_geojson()


@app.get("/staffordshire/stations")
def staffordshire_stations():
    try:
        now = time.time()
        if _STATIONS_CACHE["data"] is not None and (now - _STATIONS_CACHE["ts"]) < _STATIONS_TTL_SECONDS:
            return _STATIONS_CACHE["data"]

        _, bbox, _, prepared_poly = _load_staffs_plus_stoke_polygon_and_bbox()

        minLon = bbox["minLon"]
        minLat = bbox["minLat"]
        maxLon = bbox["maxLon"]
        maxLat = bbox["maxLat"]

        centre_lat = (minLat + maxLat) / 2
        centre_lon = (minLon + maxLon) / 2
        radius_km = max((maxLat - minLat) * 111, (maxLon - minLon) * 111) / 2
        radius_km = radius_km + 15

        url = f"https://environment.data.gov.uk/flood-monitoring/id/stations?lat={centre_lat}&long={centre_lon}&dist={radius_km}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()

        items = data.get("items", []) or []

        stations = []
        for s in items:
            lat = s.get("lat")
            lon = s.get("long")
            if lat is None or lon is None:
                continue

            try:
                pt = Point(float(lon), float(lat))
            except Exception:
                continue

            if prepared_poly.contains(pt):
                stations.append(
                    {
                        "id": s.get("stationReference"),
                        "label": s.get("label"),
                        "lat": lat,
                        "long": lon,
                    }
                )

        out = {"count": len(stations), "stations": stations}
        _STATIONS_CACHE["ts"] = now
        _STATIONS_CACHE["data"] = out
        return out

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/staffordshire/stations failed: {e}")


@app.get("/staffordshire/status")
def staffordshire_status(offset: int = 0, limit: int = 10):
    stations_payload = staffordshire_stations()
    stations = stations_payload.get("stations", [])

    total = len(stations)
    start = max(0, offset)
    end = min(total, start + max(1, limit))
    batch = stations[start:end]

    items_out = []
    max_workers = 10

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_fetch_station_row, st) for st in batch]
        for fut in as_completed(futures):
            try:
                items_out.append(fut.result())
            except Exception:
                pass

    items_out.sort(key=lambda x: (x.get("label") or "", x.get("id") or ""))

    return {
        "total": total,
        "offset": start,
        "limit": end - start,
        "items": items_out,
        "hasMore": end < total,
        "nextOffset": end if end < total else None,
    }
