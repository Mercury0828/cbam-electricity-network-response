"""Fetch one zone's generation and day-ahead price from Energy-Charts for a year.

Generic, resumable, atomic. Used to add importer zones to the GB borders and
exporter/importer zones for the Western Balkan borders, as the frozen border selection
rule admits them.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone

EC = "https://api.energy-charts.info"
UA = {"User-Agent": "wedge-fetch/0.1 (research)"}


def months(y, last=12):
    return [(f"{y}{m:02d}", f"{y}-{m:02d}-01",
             f"{y + 1}-01-01" if m == 12 else f"{y}-{m + 1:02d}-01") for m in range(1, last + 1)]


def fetch(url, tries=6):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and i < tries - 1:
                time.sleep(10 * (i + 1)); continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)      # energy-charts country code, e.g. nl
    ap.add_argument("--bzn", required=True)          # price bidding zone, e.g. NL
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--last-month", type=int, default=12)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--cbpf", action="store_true", help="also fetch cross-border physical flows")
    ap.add_argument("--no-price", action="store_true",
                    help="zone has no day-ahead price on Energy-Charts (e.g. BA, MK, MD)")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    a = ap.parse_args()
    raw = pathlib.Path(a.raw); man = pathlib.Path(a.manifest)
    done = set()
    if man.exists():
        for ln in man.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
                if r.get("status") == "OK": done.add(r["unit"])
            except Exception: pass
    units = []
    for lab, s, e in months(a.year, a.last_month):
        units.append((f"{a.country}_generation_{lab}", f"{EC}/public_power?country={a.country}&start={s}&end={e}"))
        if not a.no_price:
            units.append((f"{a.country}_price_{lab}", f"{EC}/price?bzn={a.bzn}&start={s}&end={e}"))
        if a.cbpf:
            units.append((f"{a.country}_cbpf_{lab}", f"{EC}/cbpf?country={a.country}&start={s}&end={e}"))
    units.append((f"{a.country}_installed", f"{EC}/installed_power?country={a.country}&time_step=yearly"))
    ok = 0
    with man.open("a", encoding="utf-8") as fh:
        for unit, url in units:
            tgt = raw / f"{unit}.json"
            if unit in done and tgt.exists():
                ok += 1; continue
            rec = {"unit": unit, "url": url, "source": "energy-charts",
                   "ts": datetime.now(timezone.utc).isoformat()}
            try:
                b = fetch(url); raw.mkdir(parents=True, exist_ok=True)
                tmp = tgt.with_suffix(".json.tmp"); tmp.write_bytes(b); os.replace(tmp, tgt)
                rec.update(status="OK", sha256=hashlib.sha256(b).hexdigest(), bytes=len(b)); ok += 1
            except Exception as ex:
                rec.update(status="FAIL", error=str(ex)[:200])
            fh.write(json.dumps(rec) + "\n"); fh.flush(); time.sleep(2)
    print(f"{a.country}: {ok}/{len(units)} units -> {raw}")
    return 0 if ok == len(units) else 1


if __name__ == "__main__":
    sys.exit(main())
