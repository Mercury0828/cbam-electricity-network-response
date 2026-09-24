"""Fetch Elexon FUELHH (half-hourly outturn by fuel type, settlement metering) per month.

Independent GB series used to REPAIR publisher dropouts in the per-type actual-generation feed
(`generation/actual/per-type`, B1620), where whole conventional blocks (gas, nuclear, biomass,
...) are reported as 0 while wind and solar remain. Resumable and atomic.
"""
from __future__ import annotations

import argparse, calendar, hashlib, json, os, pathlib, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone

URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH"


def fetch(url, tries=6):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, headers={"User-Agent": "wedge-fetch/0.1"}), timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and i < tries - 1:
                time.sleep(10 * (i + 1)); continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--last-month", type=int, default=12)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    a = ap.parse_args()
    raw = pathlib.Path(a.raw); raw.mkdir(parents=True, exist_ok=True)
    ok = n = 0
    with pathlib.Path(a.manifest).open("a", encoding="utf-8") as fh:
        for m in range(1, a.last_month + 1):
            n += 1
            last = calendar.monthrange(a.year, m)[1]
            unit = f"gb_fuelhh_{a.year}{m:02d}"
            tgt = raw / f"{unit}.json"
            if tgt.exists():
                ok += 1; continue
            # API limit: <= 7 days inclusive per request -> non-overlapping 7-day chunks
            chunks = [(d, min(d + 6, last)) for d in range(1, last + 1, 7)]
            url = f"{URL}?settlementDateFrom={a.year}-{m:02d}-01..{last:02d} (7-day chunks)"
            rec = {"unit": unit, "url": url, "source": "elexon-fuelhh",
                   "ts": datetime.now(timezone.utc).isoformat()}
            try:
                data = []
                for d0, d1 in chunks:
                    u = (f"{URL}?settlementDateFrom={a.year}-{m:02d}-{d0:02d}"
                         f"&settlementDateTo={a.year}-{m:02d}-{d1:02d}&format=json")
                    part = json.loads(fetch(u)).get("data") or []
                    if not part:
                        raise ValueError(f"empty data {d0}-{d1}")
                    data += part
                    time.sleep(0.5)
                b = json.dumps({"data": data}).encode("utf-8")
                tmp = tgt.with_suffix(".json.tmp"); tmp.write_bytes(b); os.replace(tmp, tgt)
                rec.update(status="OK", sha256=hashlib.sha256(b).hexdigest(), bytes=len(b)); ok += 1
            except Exception as ex:
                rec.update(status="FAIL", error=str(ex)[:200])
            fh.write(json.dumps(rec) + "\n"); fh.flush(); time.sleep(1)
    print(f"FUELHH {a.year}: {ok}/{n} months -> {raw}")
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
