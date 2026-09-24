"""Fetch Elexon REMIT messages (2019-2026) and keep those on interconnectors (A52 outage classification).

Endpoint: https://data.elexon.co.uk/bmrs/api/v1/datasets/REMIT (publish-time windows of at most one day, no key).
Kept: every message that is not a production unavailability, plus any message whose asset, unit, participant or
fuel type refers to an interconnector. Output: data/raw/remit/remit_ic.jsonl (one message per line), and a log of
days that failed.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import pathlib
import time
import urllib.request
from datetime import date, timedelta

OUT = pathlib.Path("data/raw/remit")
URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/REMIT?publishDateTimeFrom={a}T00:00Z&publishDateTimeTo={b}T00:00Z&format=json"
KEYS = ("interconn", "britned", "nemo", "ifa", "eleclink", "nsl", "viking", "moyle", "ewic", "greenlink", "i_")


def keep(x):
    if x.get("eventType") != "Production unavailability":
        return True
    s = " ".join(str(x.get(k, "")) for k in ("assetId", "affectedUnit", "participantId", "fuelType")).lower()
    return any(k in s for k in KEYS)


def day(d):
    a, b = d.isoformat(), (d + timedelta(days=1)).isoformat()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(URL.format(a=a, b=b), timeout=90) as r:
                data = json.loads(r.read().decode("utf-8")).get("data", [])
            return d, [x for x in data if keep(x)], None
        except Exception as e:                                                # noqa: BLE001
            err = str(e)
            time.sleep(2 + 3 * attempt)
    return d, [], err


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    days = [date(2019, 1, 1) + timedelta(days=i) for i in range((date(2026, 9, 1) - date(2019, 1, 1)).days)]
    done = set()
    f = OUT / "remit_ic.jsonl"
    prog = OUT / "days_done.txt"
    if prog.exists():
        done = set(prog.read_text().split())
    todo = [d for d in days if d.isoformat() not in done]
    fails = []
    with cf.ThreadPoolExecutor(8) as ex, open(f, "a", encoding="utf-8") as fo, open(prog, "a") as fp:
        for n, (d, rows, err) in enumerate(ex.map(day, todo)):
            if err:
                fails.append((d.isoformat(), err))
                continue
            for x in rows:
                fo.write(json.dumps(x) + "\n")
            fp.write(d.isoformat() + "\n")
            if n % 100 == 0:
                print(d, len(rows), flush=True)
    (OUT / "failed_days.json").write_text(json.dumps(fails), encoding="utf-8")
    print("done; failed days:", len(fails))


if __name__ == "__main__":
    main()
