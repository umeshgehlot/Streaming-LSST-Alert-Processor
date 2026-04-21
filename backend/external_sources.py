import csv
import json
import os
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from io import StringIO
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# SOTA INTEGRATION (MODULARIZED)
try:
    from src.sota.sdk import SotaSDK as SotaExternalService
    SOTA_SDK_AVAILABLE = True
except ImportError:
    SOTA_SDK_AVAILABLE = False
    logging.warning("src.sota.sdk not found. Falling back to basic urllib.")


def fetch_nasa_fireball_csv(recent_years: int = 2) -> tuple[str, bytes]:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=365 * recent_years)
    params = urlencode(
        {
            "date-min": start.strftime("%Y-%m-%d"),
            "date-max": now.strftime("%Y-%m-%d"),
            "sort": "date",
        }
    )
    url = f"https://ssd-api.jpl.nasa.gov/fireball.api?{params}"
    with urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    fields = payload.get("fields", [])
    data = payload.get("data", [])
    if len(fields) == 0 or len(data) == 0:
        raise ValueError("No fireball records returned from NASA API")
    field_index = {field_name: index for index, field_name in enumerate(fields)}
    if "date" not in field_index:
        raise ValueError("NASA response missing date field")
    impact_index = field_index.get("impact-e")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in data:
        date_value = str(row[field_index["date"]])[:10]
        impact_value = 0.0
        if impact_index is not None and row[impact_index] is not None:
            try:
                impact_value = float(row[impact_index])
            except Exception:
                impact_value = 0.0
        grouped[date_value].append(impact_value)
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["time", "flux"])
    for day in sorted(grouped.keys()):
        impacts = grouped[day]
        if len(impacts) == 0:
            flux = 0.0
        else:
            total_impact = sum(impacts)
            flux = total_impact if total_impact > 0 else float(len(impacts))
        writer.writerow([day, round(flux, 8)])
    filename = f"nasa_fireball_{start.strftime('%Y%m%d')}_{now.strftime('%Y%m%d')}.csv"
    return filename, buffer.getvalue().encode("utf-8")


def fetch_live_transient_stream(limit: int = 20) -> list[dict]:
    """
    Expert-level stream fetch via ALeRCE official client.
    """
    if SOTA_SDK_AVAILABLE:
        try:
            return SotaExternalService.fetch_alerce_stream(limit=limit)
        except Exception as e:
            logging.error(f"SOTA Stream fetch failed: {e}. Falling back to manual Antares...")
            
    external = _fetch_external_transient_stream(limit=limit)
    if len(external) > 0:
        return external
    return _build_synthetic_transient_stream(limit=limit)


def _build_synthetic_transient_stream(limit: int = 20) -> list[dict]:
    now = datetime.now(timezone.utc)
    events = []
    for index in range(max(1, min(100, limit))):
        events.append(
            {
                "event_id": f"ZTF-LSST-{now.strftime('%Y%m%d')}-{index:04d}",
                "survey": "ZTF" if index % 2 == 0 else "LSST",
                "ra": round((index * 17.23) % 360, 4),
                "dec": round(-60 + (index * 3.91) % 120, 4),
                "magnitude": round(14.0 + (index % 10) * 0.31, 3),
                "timestamp": now.isoformat(),
                "source": "synthetic-fallback",
            }
        )
    return events


def _fetch_external_transient_stream(limit: int = 20) -> list[dict]:
    base_url = os.getenv(
        "ASTRO_ALERT_STREAM_URL",
        "https://antares.noirlab.edu/api/v1/alerts/recent",
    )
    query = urlencode({"limit": max(1, min(100, int(limit)))})
    target = f"{base_url}?{query}" if "?" not in base_url else f"{base_url}&{query}"
    headers = {"User-Agent": "astro-anomaly-platform/1.0", "Accept": "application/json"}
    token = os.getenv("ASTRO_ALERT_STREAM_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        request = Request(target, headers=headers)
        with urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    rows = payload if isinstance(payload, list) else payload.get("events", [])
    normalized: list[dict] = []
    for index, row in enumerate(rows[: max(1, min(100, int(limit)))]):
        if not isinstance(row, dict):
            continue
        candidates = row.get("candidate", {}) if isinstance(row.get("candidate"), dict) else {}
        event_id = str(
            row.get("event_id")
            or row.get("objectId")
            or row.get("locus_id")
            or row.get("locusId")
            or row.get("ztf_object_id")
            or f"external-{index:04d}"
        )
        ra = candidates.get("ra", row.get("ra", 0.0))
        dec = candidates.get("dec", row.get("dec", 0.0))
        magnitude = candidates.get("magpsf", row.get("magnitude", 0.0))
        survey = str(row.get("survey") or row.get("stream") or row.get("source") or "external")
        timestamp = str(
            row.get("timestamp")
            or row.get("alert_timestamp")
            or candidates.get("jd")
            or datetime.now(timezone.utc).isoformat()
        )
        try:
            normalized.append(
                {
                    "event_id": event_id,
                    "survey": survey,
                    "ra": float(ra),
                    "dec": float(dec),
                    "magnitude": float(magnitude),
                    "timestamp": timestamp,
                    "source": base_url,
                }
            )
        except Exception:
            continue
    return normalized


def fetch_cross_match_metadata(object_name: str) -> dict:
    """
    Expert-level metadata resolution via Astroquery.Simbad.
    """
    query = object_name.strip()
    if not query:
        raise ValueError("object_name is required")
        
    if SOTA_SDK_AVAILABLE:
        try:
            res = SotaExternalService.resolve_object(name=query)
            if res.get("resolved"):
                return {
                    "object_name": query,
                    "simbad": res,
                    "vizier": {
                        "catalog_hint": "I/355/gaiadr3",
                        "query_url": f"https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=I/355/gaiadr3&-c={query}",
                        "status": "Resolved via Astroquery"
                    }
                }
        except Exception as e:
            logging.error(f"SOTA Resolution failed: {e}. Falling back to manual Sesame...")

    # Original basic resolution (Sesame manual parsing)
    simbad_url = "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV?" + query
    # ... (rest of search/match logic below)
    try:
        request = Request(simbad_url, headers={"User-Agent": "astro-anomaly-platform/1.0"})
        with urlopen(request, timeout=15) as response:
            payload = response.read().decode("utf-8", errors="ignore")
    except URLError:
        payload = ""
    aliases = []
    ra = None
    dec = None
    for raw in payload.splitlines():
        line = raw.strip()
        if line.startswith("%J"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    ra = float(parts[1])
                    dec = float(parts[2])
                except Exception:
                    ra = None
                    dec = None
        if line.startswith("%I"):
            aliases.append(line[2:].strip())
    return {
        "object_name": query,
        "simbad": {
            "resolved": len(payload) > 0,
            "aliases": aliases[:20],
            "ra_deg": ra,
            "dec_deg": dec,
        },
        "vizier": {
            "catalog_hint": "I/355/gaiadr3",
            "query_url": f"https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=I/355/gaiadr3&-c={query}",
        },
    }
