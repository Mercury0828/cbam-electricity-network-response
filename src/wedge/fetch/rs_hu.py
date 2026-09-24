"""Fetch RS -> HU from Energy-Charts, a full calendar year by default.

Resumable: every completed unit is recorded in a manifest, writes are
atomic (temp file then rename), and a restart skips what is already complete unless
--fresh is passed.

Energy-Charts (Fraunhofer ISE) republishes ENTSO-E data without a token. This is the same
acquisition route the predecessor paper S0 used;
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

BASE = "https://api.energy-charts.info"
UA = {"User-Agent": "wedge-fetch/0.1 (research)"}

def months(start_year: int, n: int = 12):
    """(label, start, end) per calendar month. Energy-Charts is chunked monthly so a
    failure costs one month, not a year, and the manifest can resume mid-year."""
    out = []
    for m in range(1, n + 1):
        a = f"{start_year}-{m:02d}-01"
        b = (f"{start_year + 1}-01-01" if m == 12 else f"{start_year}-{m + 1:02d}-01")
        out.append((f"{start_year}{m:02d}", a, b))
    return out


def build_units(year: int):
    u = []
    for lab, a, b in months(year):
        u += [
            (f"rs_generation_{lab}", f"{BASE}/public_power?country=rs&start={a}&end={b}"),
            (f"hu_generation_{lab}", f"{BASE}/public_power?country=hu&start={a}&end={b}"),
            (f"rs_price_{lab}",      f"{BASE}/price?bzn=RS&start={a}&end={b}"),
            (f"hu_price_{lab}",      f"{BASE}/price?bzn=HU&start={a}&end={b}"),
            (f"rs_cbpf_{lab}",       f"{BASE}/cbpf?country=rs&start={a}&end={b}"),
        ]
    u += [("rs_installed", f"{BASE}/installed_power?country=rs&time_step=yearly"),
          ("hu_installed", f"{BASE}/installed_power?country=hu&time_step=yearly")]
    return u


def fetch(url: str, tries: int = 5) -> bytes:
    """GET with backoff on 429. Energy-Charts rate-limits bursts."""
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < tries - 1:
                wait = 10 * (i + 1)
                print(f"    rate-limited, waiting {wait}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


def write_atomic(path: pathlib.Path, data: bytes) -> str:
    """Temp file then rename, so an interrupted run never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch RS->HU December 2025")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--raw", default="data/raw/rs_hu_2025")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()

    raw = pathlib.Path(a.raw)
    man = pathlib.Path(a.manifest)
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

    U = build_units(a.year)
    print(f"RS->HU {a.year}   ({len(U)} units, {len(done)} already complete)")
    ok = 0
    with man.open("a", encoding="utf-8") as fh:
        for unit, url in U:
            target = raw / f"{unit}.json"
            if unit in done and target.exists():
                print(f"  {unit:16s} SKIP (manifest)")
                ok += 1
                continue
            rec = {"unit": unit, "url": url, "source": "energy-charts",
                   "ts": datetime.now(timezone.utc).isoformat()}
            try:
                body = fetch(url)
                sha = write_atomic(target, body)
                payload = json.loads(body)
                n = len(payload.get("unix_seconds") or payload.get("time") or [])
                rec.update(status="OK", sha256=sha, bytes=len(body), n_timestamps=n)
                print(f"  {unit:16s} OK   {len(body):>8} bytes  n={n}")
                ok += 1
            except Exception as e:
                rec.update(status="FAIL", error=str(e)[:200])
                print(f"  {unit:16s} FAIL {str(e)[:70]}")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            time.sleep(2)  # be polite to a free public API

    print(f"\n{ok}/{len(U)} units complete -> {raw}")
    return 0 if ok == len(U) else 1


if __name__ == "__main__":
    sys.exit(main())
