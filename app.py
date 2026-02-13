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
from shapely.ops import unary_union

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import base64
import secrets
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

import time
import json
import urllib.parse
import urllib.request
from typing import Dict, Any
from fastapi import HTTPException

app = FastAPI()

# ----------------------------
# Simple Basic Auth protection
# ----------------------------
DASH_USER = os.getenv("DASH_USER", "")
DASH_PASS = os.getenv("DASH_PASS", "")
AUTH_ENABLED = bool(DASH_USER and DASH_PASS)

AUTH_SKIP_PATHS = {
    "/health",
    "/favicon.ico",
    "/apple-touch-icon.png",
    "/apple-touch-icon-precomposed.png",
}

class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if not AUTH_ENABLED:
            return await call_next(request)

        path = request.url.path
        if path in AUTH_SKIP_PATHS or path.startswith("/static"):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Basic "):
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Staffordshire Flood Intelligence"'},
                content="Authentication required",
            )

        try:
            b64 = auth.split(" ", 1)[1].strip()
            decoded = base64.b64decode(b64).decode("utf-8")
            user, pwd = decoded.split(":", 1)
        except Exception:
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Staffordshire Flood Intelligence"'},
                content="Invalid authentication",
            )

        if not (secrets.compare_digest(user, DASH_USER) and secrets.compare_digest(pwd, DASH_PASS)):
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Staffordshire Flood Intelligence"'},
                content="Invalid username or password",
            )

        return await call_next(request)

app.add_middleware(BasicAuthMiddleware)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")
# Use this file name in your repo
BOUNDARY_GEOJSON_PATH = BASE_DIR / "staffordshire_boundary.geojson"

# ----------------------------
# Simple in-memory cache
# ----------------------------
_BOUNDARY_CACHE = {
    "ts": 0.0,
    "poly": None,
    "bbox": None,
}
_BOUNDARY_TTL_SECONDS = 3600  # 1 hour

_STATIONS_CACHE = {"ts": 0.0, "data": None}
_STATIONS_TTL_SECONDS = 600  # 10 minutes

_MEASURES_CACHE = {}  # station_id -> {"ts": float, "items": list}
_MEASURES_TTL_SECONDS = 300  # 5 minutes

_STATION_META_CACHE = {}  # station_id -> {"ts": float, "meta": dict}
_STATION_META_TTL_SECONDS = 3600  # 1 hour

# Catchment convex hulls
_CATCHMENTS_CACHE = {"ts": 0.0, "geojson": None}
_CATCHMENTS_TTL_SECONDS = 21600  # 6 hour

# EA flood warnings polygons
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


def _load_boundary_poly_and_bbox():
    now = time.time()
    if (
        _BOUNDARY_CACHE["poly"] is not None
        and _BOUNDARY_CACHE["bbox"] is not None
        and (now - _BOUNDARY_CACHE["ts"]) < _BOUNDARY_TTL_SECONDS
    ):
        return _BOUNDARY_CACHE["poly"], _BOUNDARY_CACHE["bbox"]

    try:
        gj = json.loads(BOUNDARY_GEOJSON_PATH.read_text(encoding="utf-8"))
        feats = gj.get("features", []) or []
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read boundary GeoJSON at {BOUNDARY_GEOJSON_PATH}: {e}",
        )

    geoms = []
    for f in feats:
        g = f.get("geometry")
        if not g:
            continue
        try:
            geoms.append(shape(g))
        except Exception:
            continue

    if not geoms:
        raise HTTPException(status_code=404, detail="Boundary GeoJSON contained no usable geometries")

    boundary = unary_union(geoms)

    try:
        if hasattr(boundary, "is_valid") and not boundary.is_valid:
            boundary = boundary.buffer(0)
    except Exception:
        pass

    if boundary.is_empty:
        raise HTTPException(status_code=404, detail="Boundary geometry was empty")

    minx, miny, maxx, maxy = boundary.bounds

    # Sanity check: if the file is still in British National Grid metres, it will look like 300000, 400000, etc
    # This app expects WGS84 lon, lat for all point checks and the EA API bbox call
    if abs(minx) > 180 or abs(maxx) > 180 or abs(miny) > 90 or abs(maxy) > 90:
        raise HTTPException(
            status_code=500,
            detail="Boundary GeoJSON does not look like WGS84 lon, lat. Re-export it as EPSG:4326 (WGS84).",
        )

    bbox = {"minLon": minx, "minLat": miny, "maxLon": maxx, "maxLat": maxy}

    _BOUNDARY_CACHE["ts"] = now
    _BOUNDARY_CACHE["poly"] = boundary
    _BOUNDARY_CACHE["bbox"] = bbox

    return boundary, bbox


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

    _, bbox = _load_boundary_poly_and_bbox()
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

    def _extract_geom(pj: dict):
        # FeatureCollection
        if isinstance(pj, dict) and pj.get("type") == "FeatureCollection":
            feats = pj.get("features") or []
            if feats and isinstance(feats[0], dict):
                g = feats[0].get("geometry")
                if isinstance(g, dict) and g.get("type") and g.get("coordinates"):
                    return g

        # Single Feature
        if isinstance(pj, dict) and pj.get("type") == "Feature":
            g = pj.get("geometry")
            if isinstance(g, dict) and g.get("type") and g.get("coordinates"):
                return g

        # Bare geometry
        if isinstance(pj, dict) and pj.get("type") in ("Polygon", "MultiPolygon") and pj.get("coordinates"):
            return pj

        return None

    features = []
    for it in items:
        fa = it.get("floodArea") or {}
        poly = fa.get("polygon")
        if not poly:
            continue

        geom = None

        # Sometimes EA returns geometry directly
        if isinstance(poly, dict) and poly.get("type") and poly.get("coordinates"):
            geom = poly

        # Usually EA returns a URL to the polygon resource
        elif isinstance(poly, str) and poly.startswith("http"):
            try:
                pr = requests.get(poly, timeout=20)
                if pr.status_code == 200:
                    geom = _extract_geom(pr.json())
            except Exception:
                geom = None

        if not geom:
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
def _fetch_ea_flood_summary_staffordshire() -> Dict[str, Any]:
    # EA Flood Monitoring API
    # Severity levels:
    # 1 = Severe Flood Warning
    # 2 = Flood Warning
    # 3 = Flood Alert
    # 4 = Warning no longer in force

    base = "https://environment.data.gov.uk/flood-monitoring/id/floods"
    params = {
        "county": "Staffordshire",
        "min-severity": "3",
        "_limit": "500",
    }

    url = f"{base}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    with urllib.request.urlopen(req, timeout=6) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    items = payload.get("items", []) or []

    counts = {
        "severe": 0,
        "warning": 0,
        "alert": 0,
        "total": 0,
    }

    for it in items:
        sev = it.get("severityLevel")
        if sev == 1:
            counts["severe"] += 1
        elif sev == 2:
            counts["warning"] += 1
        elif sev == 3:
            counts["alert"] += 1

    counts["total"] = counts["severe"] + counts["warning"] + counts["alert"]

    return {
        "county": "Staffordshire",
        "counts": counts,
        "source": "Environment Agency Flood Monitoring API",
        "fetchedAtEpoch": int(time.time()),
    }


@app.get("/api/flood-warnings-summary")
def get_flood_warnings_summary() -> Dict[str, Any]:
    # Cache for 60 seconds to avoid hammering EA
    now = time.time()
    ttl_seconds = 60

    cached = _EA_FLOOD_SUMMARY_CACHE.get("data")
    ts = float(_EA_FLOOD_SUMMARY_CACHE.get("ts") or 0.0)

    if cached and (now - ts) < ttl_seconds:
        return cached

    try:
        data = _fetch_ea_flood_summary_staffordshire()
    except Exception as e:
        # Fail safe: return cached if available
        if cached:
            return cached
        raise HTTPException(status_code=502, detail=f"EA flood warning summary unavailable: {e}")

    _EA_FLOOD_SUMMARY_CACHE["data"] = data
    _EA_FLOOD_SUMMARY_CACHE["ts"] = now
    return data
