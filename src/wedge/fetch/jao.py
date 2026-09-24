"""Fetch JAO explicit capacity-auction results for the GB links (A36/A37), both directions.

Corridors: BritNed (BDL), Nemo Link (NLL), IFA (IF1), IFA2 (IF2), ElecLink (EL1), Viking (VKL),
GB->EU and EU->GB. Horizons: Daily (hourly products) and Monthly. 2025-01-01 .. 2026-09-01 in
chunks of <= 30 days (API limit 31). Resumable, atomic, logged in data/manifest.jsonl.

🔴 Credential discipline (as D-003 for ENTSO-E): the personal token is read from the environment
variable JAO_API_TOKEN (or the Windows user registry) and sent only as the AUTH_API_KEY header. It is
never written to disk, to the manifest, to logs or to any artefact.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone

API = "https://api.jao.eu/OWSMP/getauctions"
CORRIDORS = [f"{p}-{a}-{b}" for p, (a, b) in
             [("BDL", ("GB", "NL")), ("NLL", ("GB", "BE")), ("IF1", ("GB", "FR")),
              ("IF2", ("GB", "FR")), ("EL1", ("GB", "FR")), ("VKL", ("GB", "D1"))]
             for a, b in ((a, b), (b, a))]
HORIZONS = ("Daily", "Monthly")


def token() -> str:
    t = os.environ.get("JAO_API_TOKEN")
    if not t and sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            t = winreg.QueryValueEx(k, "JAO_API_TOKEN")[0]
    if not t:
        raise SystemExit("JAO_API_TOKEN not set")
    return t


def fetch(url: str, tok: str, tries: int = 6) -> bytes | None:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"AUTH_API_KEY": tok,
                                                       "User-Agent": "wedge-fetch/0.1"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 400:
                body = e.read()
                if b"No Data" in body:
                    return b"[]"
                raise
            if e.code in (429, 500, 502, 503) and i < tries - 1:
                time.sleep(10 * (i + 1))
                continue
            raise


def chunks(start: date, end: date, days: int = 30):
    d = start
    while d < end:
        e = min(d + timedelta(days=days), end)
        yield d, e
        d = e


def main():
    tok = token()
    raw = pathlib.Path("data/raw/jao")
    raw.mkdir(parents=True, exist_ok=True)
    man = pathlib.Path("data/manifest.jsonl")
    ok = n = 0
    with man.open("a", encoding="utf-8") as fh:
        for hz in HORIZONS:
            for c in CORRIDORS:
                for a, b in chunks(date(2025, 1, 1), date(2026, 9, 1)):
                    n += 1
                    unit = f"jao_{hz}_{c}_{a:%Y%m%d}"
                    tgt = raw / f"{unit}.json"
                    if tgt.exists():
                        ok += 1
                        continue
                    url = f"{API}?horizon={hz}&corridor={c}&fromdate={a}&todate={b}&shadow=false"
                    rec = {"unit": unit, "url": url, "source": "jao-auction-api",
                           "ts": datetime.now(timezone.utc).isoformat()}
                    try:
                        body = fetch(url, tok)
                        tmp = tgt.with_suffix(".json.tmp")
                        tmp.write_bytes(body)
                        os.replace(tmp, tgt)
                        rec.update(status="OK", sha256=hashlib.sha256(body).hexdigest(),
                                   bytes=len(body))
                        ok += 1
                    except Exception as ex:                      # token never in the message
                        rec.update(status="FAIL", error=type(ex).__name__ + ": " +
                                   str(ex).replace(tok, "<redacted>")[:200])
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    time.sleep(0.5)
    print(f"JAO: {ok}/{n} units -> {raw}")
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
