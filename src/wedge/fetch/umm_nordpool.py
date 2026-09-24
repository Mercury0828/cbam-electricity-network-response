"""Fetch transmission-unavailability urgent market messages (REMIT UMMs, message type 3) from the public Nord Pool UMM
platform API, for confirming interconnector outages.

Months of publication date 2018-06 .. 2026-09, latest version of each message (the API default excludes outdated
versions). No credentials are needed. Output: data/raw/umm/nordpool_umm_transmission.jsonl (one message per line) and
a line per month in data/manifest.jsonl.
Usage: python src/wedge/fetch/umm_nordpool.py
"""
from __future__ import annotations

import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone

API = "https://ummapi.nordpoolgroup.com/messages"
LIMIT = 2000


def get(params, tries=5):
    url = API + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wedge-fetch/0.1", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError) as e:                   # noqa: PERF203
            if i == tries - 1:
                raise
            time.sleep(10 * (i + 1))


def months(a: date, b: date):
    y, m = a.year, a.month
    while (y, m) < (b.year, b.month):
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        yield date(y, m, 1), date(ny, nm, 1)
        y, m = ny, nm


def main():
    out = pathlib.Path("data/raw/umm/nordpool_umm_transmission.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.part")
    seen, n = set(), 0
    man = pathlib.Path("data/manifest.jsonl")
    with tmp.open("w", encoding="utf-8") as fh, man.open("a", encoding="utf-8") as mf:
        for a, b in months(date(2018, 6, 1), date(2026, 10, 1)):
            skip, got = 0, 0
            while True:
                j = get({"limit": LIMIT, "skip": skip, "messageTypes": 3,
                         "publicationStartDate": f"{a:%Y-%m-%d}T00:00:00Z", "publicationStopDate": f"{b:%Y-%m-%d}T00:00:00Z"})
                items = j.get("items") or []
                for it in items:
                    k = (it.get("messageId"), it.get("version"))
                    if k in seen:
                        continue
                    seen.add(k)
                    fh.write(json.dumps(it, ensure_ascii=False) + "\n")
                    n += 1
                got += len(items)
                skip += len(items)
                if not items or skip >= (j.get("total") or 0):
                    break
            mf.write(json.dumps({"source": "nordpool_umm", "unit": f"umm_type3_{a:%Y%m}", "messages": got,
                                 "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
            print(f"{a:%Y-%m}", got, flush=True)
            time.sleep(1)
    tmp.replace(out)
    print("messages", n)


if __name__ == "__main__":
    main()
