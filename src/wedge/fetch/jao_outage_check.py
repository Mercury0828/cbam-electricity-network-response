"""Fetch JAO daily explicit-auction results on the Serbia-Hungary border around the zero-flow spells of the RS-HU
outage test (A62), both directions, for measuring how many zero-flow hours had zero offered capacity.

Corridors RS-HU and HU-RS, horizon Daily, windows of at most 30 days covering each spell from two days before to two
days after. Resumable and atomic like jao.py; every request is logged in data/manifest.jsonl.
Credential discipline as in jao.py: the token is read from JAO_API_TOKEN (environment or Windows user registry), sent
only as the AUTH_API_KEY header, and never written to disk, the manifest, logs or artefacts.
Usage: python src/wedge/fetch/jao_outage_check.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.fetch.jao import API, chunks, fetch, token   # noqa: E402

SPELLS = [("2023-08-12", "2023-09-04"), ("2024-09-09", "2024-09-30"), ("2025-08-11", "2025-08-13")]
CORRIDORS = ("RS-HU", "HU-RS")


def main():
    tok = token()
    raw = pathlib.Path("data/raw/jao")
    raw.mkdir(parents=True, exist_ok=True)
    man = pathlib.Path("data/manifest.jsonl")
    ok = n = 0
    with man.open("a", encoding="utf-8") as fh:
        for s, e in SPELLS:
            a0 = date.fromisoformat(s) - timedelta(days=2)
            b0 = date.fromisoformat(e) + timedelta(days=3)
            for c in CORRIDORS:
                for a, b in chunks(a0, b0):
                    n += 1
                    unit = f"jao_Daily_{c}_{a:%Y%m%d}"
                    tgt = raw / f"{unit}.json"
                    if tgt.exists():
                        ok += 1
                        continue
                    url = f"{API}?horizon=Daily&corridor={c}&fromdate={a}&todate={b}&shadow=false"
                    rec = {"unit": unit, "url": url, "source": "jao-auction-api",
                           "ts": datetime.now(timezone.utc).isoformat()}
                    try:
                        body = fetch(url, tok)
                        tmp = tgt.with_suffix(".json.tmp")
                        tmp.write_bytes(body)
                        os.replace(tmp, tgt)
                        rec.update(status="OK", sha256=hashlib.sha256(body).hexdigest(), bytes=len(body))
                        ok += 1
                    except Exception as ex:                      # token never in the message
                        rec.update(status="FAIL", error=type(ex).__name__ + ": " + str(ex).replace(tok, "<redacted>")[:200])
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    time.sleep(0.5)
    print(f"JAO RS-HU: {ok}/{n} units -> {raw}")
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
