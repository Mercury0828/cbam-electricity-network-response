"""Availability probe for the sources that need no credentials.

Covers the GB side (Elexon BMRS Insights) and the FR side (ODRE / eco2mix), which
together carry the frozen `GB -> FR` pair without any
ENTSO-E access.

Like `entsoe_probe`, this classifies on the PARSED payload, never on the HTTP status
code alone. Guide amendment A5 records why: one ENTSO-E host returns HTTP 200 with an
HTML page for every query, which a status-only check scores as success.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request
from datetime import datetime, timezone

UA = {"User-Agent": "wedge-probe/0.1 (research; contact via the repository)"}

ELEXON = "https://data.elexon.co.uk/bmrs/api/v1"
ODRE = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
# Fraunhofer ISE Energy-Charts republishes ENTSO-E data with NO token.
# This is the route the predecessor paper (S0) used;
EC = "https://api.energy-charts.info"

# Each probe: (side, series, url, extractor)
def _checks(frm: str, to: str) -> list[tuple]:
    return [
        ("GB", "generation_per_type",
         f"{ELEXON}/generation/actual/per-type?from={frm}T00:00Z&to={frm}T06:00Z",
         lambda d: (len(d.get("data", [])),
                    f"{len(d['data'][0]['data'])} psrTypes, settlement-period (30 min)"
                    if d.get("data") else "")),
        ("GB", "generation_5min",
         f"{ELEXON}/datasets/FUELINST?publishDateTimeFrom={frm}T00:00Z&publishDateTimeTo={frm}T01:00Z",
         lambda d: (len(d.get("data", [])), "FUELINST, 5-minute")),
        ("GB", "day_ahead_price",
         f"{ELEXON}/balancing/pricing/market-index?from={frm}T00:00Z&to={frm}T06:00Z",
         lambda d: (len(d.get("data", [])),
                    "MID " + ",".join(sorted({r["dataProvider"] for r in d.get("data", [])})))),
        ("GB->FR", "interconnector_flow",
         f"{ELEXON}/generation/outturn/interconnectors",
         lambda d: (len(d.get("data", [])),
                    "INTOUTHH per-link: " + ",".join(sorted({
                        r["interconnectorName"] for r in d.get("data", [])})[:6]))),
        ("FR", "generation_per_type_def",
         f"{ODRE}/eco2mix-national-cons-def/records?limit=1&order_by=-date_heure",
         lambda d: (d.get("total_count", 0),
                    "15-min consolidated; latest " + str(
                        (d.get("results") or [{}])[0].get("date_heure")))),
        ("FR", "generation_per_type_realtime",
         f"{ODRE}/eco2mix-national-tr/records?limit=1&order_by=-date_heure",
         lambda d: (d.get("total_count", 0),
                    "15-min realtime; latest " + str(
                        (d.get("results") or [{}])[0].get("date_heure")))),
        ("RS", "generation_per_type",
         f"{EC}/public_power?country=rs&start={frm}&end={frm}",
         lambda d: (len(d.get("production_types", [])),
                    "Energy-Charts, incl " + ", ".join(
                        x["name"] for x in d.get("production_types", [])
                        if "lignite" in x["name"].lower() or "gas" in x["name"].lower()))),
        ("HU", "generation_per_type",
         f"{EC}/public_power?country=hu&start={frm}&end={frm}",
         lambda d: (len(d.get("production_types", [])), "Energy-Charts")),
        ("RS", "day_ahead_price",
         f"{EC}/price?bzn=RS&start={frm}&end={frm}",
         lambda d: (len(d.get("price", [])), str(d.get("license_info", ""))[:60])),
        ("HU", "day_ahead_price",
         f"{EC}/price?bzn=HU&start={frm}&end={frm}",
         lambda d: (len(d.get("price", [])), str(d.get("license_info", ""))[:60])),
        ("RS->HU", "physical_flow",
         f"{EC}/cbpf?country=rs&start={frm}&end={frm}",
         lambda d: (len(d.get("countries", [])),
                    "counterparts: " + ", ".join(x.get("name", "?") for x in d.get("countries", [])))),
        ("UA", "day_ahead_price",
         f"{EC}/price?bzn=UA-IPS&start={frm}&end={frm}",
         lambda d: (len(d.get("price", [])),
                    "UA-IPS; CHECK CURRENCY (looks like UAH not EUR); license: "
                    + str(d.get("license_info", ""))[:40])),
        ("UA", "generation_per_type",
         f"{EC}/public_power?country=ua&start={frm}&end={frm}",
         lambda d: (0, "NOT AVAILABLE on Energy-Charts at any country code tried")),
        ("FR", "day_ahead_price_ec",
         f"{EC}/price?bzn=FR&start={frm}&end={frm}",
         lambda d: (len(d.get("price", [])), str(d.get("license_info", ""))[:60])),
        ("FR", "day_ahead_price",
         f"{ODRE}?where=search(title%2C%27prix%27)&limit=5",
         lambda d: (0, "NO hourly price dataset on ODRE ("
                       f"{d.get('total_count', 0)} title matches, none hourly spot)")),
    ]


def run(frm: str, to: str, manifest: pathlib.Path, fresh: bool) -> int:
    done = set()
    if manifest.exists() and not fresh:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done.add((r.get("side"), r.get("series")))
                except json.JSONDecodeError:
                    pass
    elif fresh and manifest.exists():
        manifest.unlink()

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as fh:
        for side, series, url, extract in _checks(frm, to):
            if (side, series) in done:
                print(f"  {side:7s} {series:28s} SKIP (already done)")
                continue
            rec = {"source": "open", "side": side, "series": series,
                   "ts": datetime.now(timezone.utc).isoformat(), "url": url}
            t0 = time.time()
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=90) as r:
                    raw = r.read()
                    ctype = r.headers.get("Content-Type", "")
                if "json" not in ctype:
                    rec.update(http=200, status="NOT_JSON", detail=ctype[:60], n=0)
                else:
                    d = json.loads(raw)
                    n, detail = extract(d)
                    rec.update(http=200, status=("OK" if n else "EMPTY"),
                               detail=str(detail)[:120], n=n, bytes=len(raw))
            except Exception as e:
                rec.update(http=0, status="ERROR", detail=str(e)[:160], n=0)
            rec["secs"] = round(time.time() - t0, 1)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  {side:7s} {series:28s} {rec['status']:9s} n={rec['n']:<7} {rec['detail'][:70]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Availability probe: credential-free sources")
    ap.add_argument("--from", dest="frm", default="2026-03-01")
    ap.add_argument("--to", dest="to", default="2026-03-08")
    ap.add_argument("--manifest", default="data/probe_manifest_open.jsonl")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    print(f"open-source probe, window {a.frm} -> {a.to}")
    return run(a.frm, a.to, pathlib.Path(a.manifest), a.fresh)


if __name__ == "__main__":
    sys.exit(main())
