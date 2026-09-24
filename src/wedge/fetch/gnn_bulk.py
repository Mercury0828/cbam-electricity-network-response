"""Bulk raw data for the graph model (A49): 2019-01 .. 2026-08, hourly inputs for 17 zones.

Continental zones (the 15 of the earlier MG-STGNN study plus Serbia) come from Energy-Charts: generation by type,
day-ahead price of the named bidding zone and cross-border physical flows (yearly requests, --part ecy).
Great Britain comes from Elexon BMRS: generation per type (AGPT), FUELHH, Market Index Price, interconnector
outturn and initial demand outturn, in 7-day chunks.
Resumable and atomic: a unit is skipped when its file exists. Usage:
    python src/wedge/fetch/gnn_bulk.py --part ec --zones at,be --years 2019-2026
    python src/wedge/fetch/gnn_bulk.py --part gb --years 2019-2026
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

EC = "https://api.energy-charts.info"
EL = "https://data.elexon.co.uk/bmrs/api/v1"
UA = {"User-Agent": "wedge-fetch/0.2 (research)"}
RAW = pathlib.Path("data/raw/gnn")
LAST = dt.date(2026, 9, 1)
PAUSE = float(os.environ.get("EC_PAUSE", "1.0"))                      # exclusive end of the study window
ZONES = {"at": "AT", "be": "BE", "ch": "CH", "cz": "CZ", "de": "DE-LU", "dk": "DK1", "es": "ES", "fr": "FR",
         "hu": "HU", "it": "IT-North", "nl": "NL", "no": "NO2", "pl": "PL", "se": "SE3", "sk": "SK", "rs": "RS",
         # A53 boundary closure: neighbours of RS and HU, Ireland (GB links), Portugal (ES link)
         "ro": "RO", "bg": "BG", "hr": "HR", "si": "SI", "gr": "GR", "ba": "BA", "me": "ME", "mk": "MK",
         "ie": "IE-SEM", "pt": "PT"}


def get(url, tries=8):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=240) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError) as e:
            code = getattr(e, "code", None)
            if code in (400, 404):
                raise
            time.sleep(min(120, 10 * (i + 1)))
    raise RuntimeError(f"failed after {tries}: {url}")


def save(path, url, pause):
    if path.exists():
        return "skip"
    b = get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(b)
    os.replace(tmp, path)
    time.sleep(pause)
    return "ok"


def months(y):
    out = []
    for m in range(1, 13):
        a = dt.date(y, m, 1)
        b = dt.date(y + (m == 12), 1 if m == 12 else m + 1, 1)
        if a >= LAST:
            break
        out.append((f"{y}{m:02d}", a.isoformat(), b.isoformat()))
    return out


def weeks(y):
    d, end = dt.date(y, 1, 1), min(dt.date(y + 1, 1, 1), LAST)
    out = []
    while d < end:
        n = min(d + dt.timedelta(days=7), end)
        out.append((d, n))
        d = n
    return out


def run_ec_yearly(zones, years):
    """Whole-year requests (Energy-Charts returns hourly resolution for long ranges): 3 requests per zone-year."""
    for cc in zones:
        bzn = ZONES[cc]
        for y in years:
            a = f"{y}-01-01"
            b = min(dt.date(y + 1, 1, 1), LAST).isoformat()
            for kind, url in (("generation", f"{EC}/public_power?country={cc}&start={a}&end={b}"),
                              ("price", f"{EC}/price?bzn={bzn}&start={a}&end={b}"),
                              ("cbpf", f"{EC}/cbpf?country={cc}&start={a}&end={b}")):
                p = RAW / "ec_y" / cc / f"{cc}_{kind}_{y}.json"
                try:
                    save(p, url, PAUSE)
                except Exception as e:
                    print("FAIL", p.name, str(e)[:120], flush=True)
        print(cc, "done", flush=True)


def run_ec(zones, years):
    for cc in zones:
        bzn = ZONES[cc]
        for y in years:
            for lab, a, b in months(y):
                for kind, url in (("generation", f"{EC}/public_power?country={cc}&start={a}&end={b}"),
                                  ("price", f"{EC}/price?bzn={bzn}&start={a}&end={b}"),
                                  ("cbpf", f"{EC}/cbpf?country={cc}&start={a}&end={b}")):
                    p = RAW / "ec" / cc / f"{cc}_{kind}_{lab}.json"
                    try:
                        save(p, url, 1.5)
                    except Exception as e:
                        print("FAIL", p.name, str(e)[:120], flush=True)
            print(cc, y, "done", flush=True)


def run_gb(years):
    for y in years:
        for i, (a, b) in enumerate(weeks(y)):
            A, B = a.isoformat(), b.isoformat()
            units = {
                "agpt": f"{EL}/generation/actual/per-type?from={A}T00:00Z&to={B}T00:00Z",
                "fuelhh": f"{EL}/datasets/FUELHH?settlementDateFrom={A}&settlementDateTo={(b - dt.timedelta(days=1)).isoformat()}",
                "mid": f"{EL}/balancing/pricing/market-index?from={A}T00:00Z&to={B}T00:00Z",
                "ic": f"{EL}/generation/outturn/interconnectors?settlementDateFrom={A}&settlementDateTo={B}",
                "demand": f"{EL}/demand/outturn?settlementDateFrom={A}&settlementDateTo={B}",
            }
            for k, url in units.items():
                p = RAW / "gb" / str(y) / f"gb_{k}_{A}.json"
                try:
                    save(p, url, 0.5)
                except Exception as e:
                    print("FAIL", p.name, str(e)[:120], flush=True)
        print("gb", y, "done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["ec", "ecy", "gb"], required=True)
    ap.add_argument("--zones", default=",".join(ZONES))
    ap.add_argument("--years", default="2019-2026")
    a = ap.parse_args()
    y0, y1 = (int(x) for x in a.years.split("-"))
    years = list(range(y0, y1 + 1))
    if a.part == "ecy":
        run_ec_yearly(a.zones.split(","), years)
    elif a.part == "ec":
        run_ec(a.zones.split(","), years)
    else:
        run_gb(years)
    return 0


if __name__ == "__main__":
    sys.exit(main())
