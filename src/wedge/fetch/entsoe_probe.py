"""Availability probe against the ENTSO-E Transparency Platform.

This is a PROBE, not the pipeline. It answers one question per
(zone, series): does the data exist, at what resolution, from which endpoint.

Two things this module is careful about:

1. The token is read from the environment and is NEVER written to a file, a log line
   or a manifest entry. `_redact` strips it from any URL before it is recorded.
2. Every probe result is appended to a JSONL manifest and completed units are skipped
   on restart unless --fresh is passed.

The API host is `https://web-api.tp.entsoe.eu/api`. Do NOT use
`https://transparency.entsoe.eu/api`: it returns HTTP 200 with the HTML web application for any
query, which a status-code-only check records as success.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

API = "https://web-api.tp.entsoe.eu/api"

# ENTSO-E EIC area codes for the frozen border pairs
# plus the declared alternates, so a failover does not need a code lookup.
ZONES = {
    "UA": "10Y1001C--00003F",   # Ukraine (IPS/UPS)
    "HU": "10YHU-MAVIR----U",
    "RS": "10YCS-SERBIATSOV",
    "GB": "10YGB----------A",
    "FR": "10YFR-RTE------C",
    # alternates
    "SK": "10YSK-SEPS-----K",
    "RO": "10YRO-TEL------P",
    "BA": "10YBA-JPCC-----D",
    "HR": "10YHR-HEP------M",
    "BE": "10YBE----------2",
    "NL": "10YNL----------L",
}

# The frozen pairs, exporter -> EU zone.
PAIRS = [("UA", "HU"), ("RS", "HU"), ("GB", "FR")]

# What each side needs. `scope` says whether it is a single-area or a pair query.
SERIES = {
    "day_ahead_price":      dict(documentType="A44", scope="pair_same"),
    "generation_per_type":  dict(documentType="A75", processType="A16", scope="area_in"),
    "total_load":           dict(documentType="A65", processType="A16", scope="area_out"),
    "installed_capacity":   dict(documentType="A68", processType="A33", scope="area_in"),
    "physical_flow":        dict(documentType="A11", scope="pair_dir"),
    "commercial_schedule":  dict(documentType="A09", scope="pair_dir"),
}

NS = "{urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:8:0}"


def _token() -> str:
    tok = os.environ.get("ENTSOE_API_TOKEN", "").strip()
    if not tok:
        raise SystemExit(
            "ENTSOE_API_TOKEN is not set in this process.\n"
            "It may be set at user level but not inherited. Pass it via the environment."
        )
    return tok


def _redact(url: str, tok: str) -> str:
    return url.replace(tok, "<REDACTED>")


def _build(params: dict, tok: str) -> str:
    q = dict(params)
    q["securityToken"] = tok
    return f"{API}?{urllib.parse.urlencode(q)}"


def _classify(body: bytes) -> tuple[str, str, int]:
    """Return (status, detail, n_points) for a 200 response body."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        return "PARSE_ERROR", str(exc)[:160], 0
    tag = root.tag.split("}")[-1]
    if tag == "Acknowledgement_MarketDocument":
        reason = root.find(f".//{NS}Reason/{NS}text")
        if reason is None:  # namespace may differ; fall back to a local-name walk
            for el in root.iter():
                if el.tag.split("}")[-1] == "text":
                    reason = el
                    break
        return "NO_DATA", (reason.text if reason is not None else "acknowledgement")[:200], 0
    n = 0
    resolutions = set()
    for el in root.iter():
        ln = el.tag.split("}")[-1]
        if ln == "Point":
            n += 1
        elif ln == "resolution" and el.text:
            resolutions.add(el.text)
    return "OK", ",".join(sorted(resolutions)) or "?", n


def probe_one(series: str, spec: dict, exp: str, imp: str, start: str, end: str,
              tok: str, timeout: int = 90) -> dict:
    scope = spec["scope"]
    base = {k: v for k, v in spec.items() if k not in ("scope",)}
    base["periodStart"] = start
    base["periodEnd"] = end

    if scope == "pair_same":       # price is asked per area, in==out
        targets = [(exp, exp), (imp, imp)]
    elif scope == "area_in":
        targets = [(exp, None), (imp, None)]
    elif scope == "area_out":
        targets = [(exp, None), (imp, None)]
    elif scope == "pair_dir":      # exporter -> importer direction
        targets = [(exp, imp)]
    else:
        raise ValueError(scope)

    out = []
    for a, b in targets:
        p = dict(base)
        if scope == "pair_same":
            p["in_Domain"] = ZONES[a]
            p["out_Domain"] = ZONES[a]
            label = a
        elif scope == "area_in":
            p["in_Domain"] = ZONES[a]
            label = a
        elif scope == "area_out":
            p["outBiddingZone_Domain"] = ZONES[a]
            label = a
        else:
            p["out_Domain"] = ZONES[a]
            p["in_Domain"] = ZONES[b]
            label = f"{a}->{b}"

        url = _build(p, tok)
        rec = dict(series=series, target=label, ts=datetime.now(timezone.utc).isoformat(),
                   url=_redact(url, tok))
        t0 = time.time()
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                body = r.read()
            status, detail, n = _classify(body)
            rec.update(http=200, status=status, detail=detail, n_points=n,
                       bytes=len(body))
        except urllib.error.HTTPError as e:
            try:
                body = e.read()
                _, detail, _ = _classify(body)
            except Exception:
                detail = ""
            rec.update(http=e.code, status="HTTP_ERROR", detail=detail[:200],
                       n_points=0, bytes=0)
        except Exception as e:  # network/timeout
            rec.update(http=0, status="ERROR", detail=str(e)[:200], n_points=0, bytes=0)
        rec["secs"] = round(time.time() - t0, 1)
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="ENTSO-E availability probe")
    ap.add_argument("--start", default="202603010000")
    ap.add_argument("--end", default="202603080000")
    ap.add_argument("--manifest", default="data/probe_manifest.jsonl")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()

    tok = _token()
    man = pathlib.Path(a.manifest)
    man.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple] = set()
    if man.exists() and not a.fresh:
        for line in man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("window") == [a.start, a.end]:
                done.add((r["series"], r["target"]))
    if a.fresh and man.exists():
        man.unlink()

    print(f"window {a.start} -> {a.end}   (skipping {len(done)} completed units)")
    with man.open("a", encoding="utf-8") as fh:
        for exp, imp in PAIRS:
            print(f"\n=== {exp} -> {imp} ===")
            for series, spec in SERIES.items():
                for rec in probe_one(series, spec, exp, imp, a.start, a.end, tok):
                    key = (rec["series"], rec["target"])
                    if key in done:
                        print(f"  {series:22s} {rec['target']:9s} SKIP (already done)")
                        continue
                    rec["window"] = [a.start, a.end]
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    print(f"  {series:22s} {rec['target']:9s} "
                          f"http={rec['http']:3d} {rec['status']:11s} "
                          f"pts={rec['n_points']:5d} res={rec['detail'][:40]}")
    print(f"\nmanifest -> {man}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
