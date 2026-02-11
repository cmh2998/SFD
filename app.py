# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import re
import json
import requests

from shapely.geometry import Point, MultiPoint, shape, mapping
from shapely.prepared import prep
from shapely.ops import unary_union
from pyproj import Transformer

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
BOUNDARY_PATH = BASE_DIR / "county_region.GeoJSON"
TEMPLATES_INDEX = BASE_DIR / "templates" / "index.html"

# Helpful for some envs, harmless otherwise
os.environ["SHAPE_RESTORE_SHX"] = "YES"

# ----------------------------
# Caches
# ----------------------------
_AREA_CACHE = {
    "ts": 0.0,
    "poly": None,          # shapely geometry in WGS84
    "bbox": None,          # dict
    "includes": None,      # list[str]
    "boundary_geojson": None,  # FeatureCollection in WGS84 for Leaflet
}
_AREA_TTL_SECONDS = 3600

_STATIONS_CACHE = {"ts": 0.0, "data": None}
_STATIONS_TTL_SECONDS = 600

_MEASURES_CACHE = {}
_MEASURES_TTL_SECONDS = 300

_STATION_META_CACHE = {}
_STATION_META_TTL_SECONDS = 3600

_CATCHMENTS_CACHE = {"ts": 0.0, "geojson": None}
_CATCHMENTS_TTL_SECONDS = 3600

_FLOODWARN_CACHE = {"ts": 0.0, "geojson": None}
_FLOODWARN_TTL_SECONDS = 300

OFFLINE_AFTER_HOURS = 6

# BNG -> WGS84 transformer (EPSG:27700 to EPSG:4326)
_BNG_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)


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


def _get_prop_name(props):
    if not isinstance(props, dict):
        return ""
    return str(
        props.get("LAD17NM")
        or props.get("LAD22NM")
        or props.get("LAD21NM")
        or props.get("LAD20NM")
        or props.get("NAME")
        or props.get("Name")
        or props.get("name")
        or ""
    ).strip()


def _transform_coords_bng_to_wgs84(coords):
    # Handles nested coordinate arrays for Polygon/MultiPolygon
    if coords is None:
        return coords

    if isinstance(coords, (list, tuple)):
        if len(coords) == 2 and all(isinstance(x, (int, float)) for x in coords):
            x, y = coords[0], coords[1]
            lon, lat = _BNG_TO_WGS84.transform(x, y)
            return [float(lon), float(lat)]
        return [_transform_coords_bng_to_wgs84(c) for c in coords]

    return coords


def _geometry_to_wgs84(geom_obj, crs_name):
    """
    If boundary GeoJSON is EPSG:27700, transform to EPSG:4326.
    If already WGS84 or unknown, assume it is already lon/lat.
    """
    if not isinstance(geom_obj, dict):
        return None

    gtype = geom_obj.get("type")
    coords = geom_obj.get("coordinates")

    if not gtype or coords is None:
        return None

    if crs_name and "27700" in crs_name:
        coords_out = _transform_coords_bng_to_wgs84(coords)
        return {"type": gtype, "coordinates": coords_out}

    # Assume already WGS84
    return geom_obj


def _load_boundary_geojson_raw():
    try:
        return json.loads(BOUNDARY_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read boundary GeoJSON at {BOUNDARY_PATH}: {e}")


def _load_staffs_plus_stoke_polygon_and_bbox():
    now = time.time()
    if (
        _AREA_CACHE["poly"] is not None
        and _AREA_CACHE["bbox"] is not None
        and (now - _AREA_CACHE["ts"]) < _AREA_TTL_SECONDS
    ):
        return _AREA_CACHE["poly"], _AREA_CACHE["bbox"], _AREA_CACHE["includes"], _AREA_CACHE["boundary_geojson"]

    raw = _load_boundary_geojson_raw()
    features = raw.get("features") or []

    if not features:
        raise HTTPException(status_code=404, detail="Boundary GeoJSON contained no features")

    crs_name = ""
    crs = raw.get("crs") or {}
    if isinstance(crs, dict):
        props = crs.get("properties") or {}
        crs_name = str(props.get("name") or "")

    wanted = ["staffordshire", "stoke-on-trent", "stoke on trent", "stoke-on trent"]
    kept = []
    for f in features:
        props = f.get("properties") or {}
        nm = _get_prop_name(props)
        if not nm:
            continue
        low = nm.lower()
        if any(w in low for w in wanted):
            kept.append(f)

    # If the file does not contain those names, fallback to using all features
    use_features = kept if kept else features

    geoms = []
    includes = []

    out_features = []
    for f in use_features:
        geom_in = f.get("geometry")
        if not geom_in:
            continue

        geom_wgs = _geometry_to_wgs84(geom_in, crs_name)
        if not geom_wgs:
            continue

        try:
            s = shape(geom_wgs)
            if s.is_empty:
                continue
            geoms.append(s)
        except Exception:
            continue

        props = f.get("properties") or {}
        nm = _get_prop_name(props)
        if nm:
            includes.append(nm)

        out_features.append({
            "type": "Feature",
            "properties": props,
            "geometry": geom_wgs
        })

    if not geoms:
        raise HTTPException(status_code=404, detail="Boundary GeoJSON contained no usable geometries")

    area_poly = unary_union(geoms)

    try:
        if hasattr(area_poly, "is_valid") and not area_poly.is_valid:
            area_poly = area_poly.buffer(0)
    except Exception:
        pass

    minx, miny, maxx, maxy = area_poly.bounds
    bbox = {"minLon": float(minx), "minLat": float(miny), "maxLon": float(maxx), "maxLat": float(maxy)}

    boundary_geojson = {"type": "FeatureCollection", "features": out_features}

    _AREA_CACHE["ts"] = now
    _AREA_CACHE["poly"] = area_poly
    _AREA_CACHE["bbox"] = bbox
    _AREA_CACHE["includes"] = sorted(list({n for n in includes if n}))
    _AREA_CACHE["boundary_geojson"] = boundary_geojson

    return area_poly, bbox, _AREA_CACHE["includes"], boundary_geojson


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

    row["reporting"] = "Offline" if age > (OFFLINE_AFTER_HOURS * 3600) else "Live"
    row["state"] = _state_for_measure(parameter, unit_short, value)

    if measure_id:
        trend, diff, rate = _compute_trend_for_measure(measure_id)
        row["trend"] = trend
        row["trendDiff"] = diff
        row["trendRatePerHour"] = rate

    return row


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

        features.append({
            "type": "Feature",
            "properties": {"catchmentName": cn, "stationCount": len(pts)},
            "geometry": mapping(hull)
        })

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

        if not (isinstance(poly, dict) and poly.get("type") and poly.get("coordinates")):
            continue

        features.append({
            "type": "Feature",
            "properties": {
                "severity": it.get("severity"),
                "severityLevel": it.get("severityLevel"),
                "message": it.get("message"),
                "timeRaised": it.get("timeRaised"),
                "timeMessageChanged": it.get("timeMessageChanged"),
                "area": fa.get("label") or fa.get("fwdCode") or "Flood area"
            },
            "geometry": poly
        })

    out = {"type": "FeatureCollection", "features": features}
    _FLOODWARN_CACHE["ts"] = now
    _FLOODWARN_CACHE["geojson"] = out
    return out


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def home():
    try:
        return Path(TEMPLATES_INDEX).read_text(encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read templates/index.html: {e}")


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
    _, _, _, boundary_geojson = _load_staffs_plus_stoke_polygon_and_bbox()
    return boundary_geojson


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

        area_poly, bbox, _, _ = _load_staffs_plus_stoke_polygon_and_bbox()
        area_prepped = prep(area_poly)

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

        items = data.get("items", [])
        stations = []

        for s in items:
            lat = s.get("lat")
            lon = s.get("long")
            if lat is None or lon is None:
                continue

            pt = Point(float(lon), float(lat))
            if not area_prepped.contains(pt):
                continue

            stations.append({
                "id": s.get("stationReference"),
                "label": s.get("label"),
                "lat": lat,
                "long": lon,
            })

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
    with ThreadPoolExecutor(max_workers=10) as ex:
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
