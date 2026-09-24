"""Fetch the second border-month: GB -> FR, December 2025.

The RS -> HU month returned zero
determined hours because both sides are dominated by technologies with no constructible SRMC
band (mine-mouth lignite, hydro, biomass) and Serbia had no carbon price in 2025. GB -> FR is
the opposite case: gas-marginal and carbon-priced on both sides (UK ETS and EU ETS), so it
tests whether the identification failure is specific to that border or general to the method.

GB comes from Elexon BMRS (open, no token); FR from Energy-Charts (open, no token).
Resumable and atomic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

ELEXON = "https://data.elexon.co.uk/bmrs/api/v1"
EC = "https://api.energy-charts.info"
UA = {"User-Agent": "wedge-fetch/0.1 (research)"}

import datetime as _dt


def _weeks(year: int, last_month: int = 12):
    """Elexon caps window length, so GB series are fetched in 7-day chunks."""
    d = _dt.date(year, 1, 1)
    end = (_dt.date(year + 1, 1, 1) if last_month == 12
           else _dt.date(year, last_month + 1, 1))
    out = []
    while d < end:
        nxt = min(d + _dt.timedelta(days=7), end)
        out.append((d.isoformat(), nxt.isoformat()))
        d = nxt
    return out


def _months(year: int, last_month: int = 12):
    out = []
    for m in range(1, last_month + 1):
        a = f"{year}-{m:02d}-01"
        b = (f"{year + 1}-01-01" if m == 12 else f"{year}-{m + 1:02d}-01")
        out.append((f"{year}{m:02d}", a, b))
    return out


def units(year: int = 2025, last_month: int = 12):
    u = []
    for lab, a, b in _months(year, last_month):
        u += [(f"fr_generation_{lab}", f"{EC}/public_power?country=fr&start={a}&end={b}"),
              (f"fr_price_{lab}",      f"{EC}/price?bzn=FR&start={a}&end={b}")]
    for i, (a, b) in enumerate(_weeks(year, last_month)):
        u.append((f"gb_generation_{i}",
                  f"{ELEXON}/generation/actual/per-type?from={a}T00:00Z&to={b}T00:00Z"))
        u.append((f"gb_price_{i}",
                  f"{ELEXON}/balancing/pricing/market-index?from={a}T00:00Z&to={b}T00:00Z"))
        u.append((f"gb_interconnector_{i}",
                  f"{ELEXON}/generation/outturn/interconnectors"
                  f"?settlementDateFrom={a}&settlementDateTo={b}"))
    return u


def fetch(url: str, tries: int = 5) -> bytes:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and i < tries - 1:
                wait = 10 * (i + 1)
                print(f"    {e.code}, waiting {wait}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


def write_atomic(path: pathlib.Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--last-month", type=int, default=12)
    ap.add_argument("--raw", default="data/raw/gb_fr_2025")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()

    raw, man = pathlib.Path(a.raw), pathlib.Path(a.manifest)
    man.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if man.exists() and not a.fresh:
        for line in man.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    if r.get("status") == "OK":
                        done.add(r["unit"])
                except json.JSONDecodeError:
                    pass

    U = units(a.year, a.last_month)
    print(f"GB->FR {a.year}   ({len(U)} units)")
    ok = 0
    with man.open("a", encoding="utf-8") as fh:
        for unit, url in U:
            target = raw / f"{unit}.json"
            if unit in done and target.exists():
                print(f"  {unit:24s} SKIP")
                ok += 1
                continue
            rec = {"unit": unit, "url": url, "source": "elexon|energy-charts",
                   "ts": datetime.now(timezone.utc).isoformat()}
            try:
                body = fetch(url)
                sha = write_atomic(target, body)
                d = json.loads(body)
                n = len(d.get("unix_seconds") or d.get("data") or [])
                rec.update(status="OK", sha256=sha, bytes=len(body), n=n)
                print(f"  {unit:24s} OK   {len(body):>8} bytes n={n}")
                ok += 1
            except Exception as e:
                rec.update(status="FAIL", error=str(e)[:200])
                print(f"  {unit:24s} FAIL {str(e)[:60]}")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            time.sleep(1.5)

    print(f"\n{ok}/{len(U)} units complete -> {raw}")
    return 0 if ok == len(U) else 1


if __name__ == "__main__":
    sys.exit(main())
