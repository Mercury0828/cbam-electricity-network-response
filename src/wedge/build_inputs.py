"""Rebuild the processed fuel tables from the archived primary files, and verify the frozen copies.

🔴 A29: the fuel tables in data/processed/ were
assembled in-session with no builder. The primary files are now archived in data/inputs/source/
(tracked; small; public sources) and this script regenerates:

  ttf_daily_frontmonth_acer.json  TTF front-month = ACER "EU PRICE" - "LNG BENCHMARK" (ACER LNG
                                  price assessment methodology v1.1: benchmark = EU LNG price minus
                                  ICE TTF front-month settlement). Source: aegis.acer.europa.eu/
                                  terminal/price_assessments/historical_data (CSV, 2026-09-20).
  fuel_monthly_2025.json          World Bank CMO Pink Sheet, "Natural gas, Europe" ($/mmbtu) and
  fuel_monthly_2026.json          "Coal, South African" ($/mt); 2025 from the January-2026 vintage,
                                  2026 from the September-2026 vintage (vintages differ in rounding).

carbon_monthly_2026.json is a hand transcription of sourced monthly values (KOBiZE EUA table,
DESNZ/UKA, BoE and ECB FX; round carbon-prices-2026) and is archived as
data/inputs/carbon_monthly_2026.json with that provenance; it has no machine-readable source.

Usage: python src/wedge/build_inputs.py [--write]   (default: verify only)
"""
from __future__ import annotations

import csv, json, pathlib, sys

SRC = pathlib.Path("data/inputs/source")
OUT = pathlib.Path("data/processed")


def ttf():
    out = {}
    with (SRC / "acer_lng_price_assessments_2026-09-20.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                out[r["DATE"]] = round(float(r["EU PRICE (EUR/MWh)"]) -
                                       float(r["LNG BENCHMARK (EUR/MWh)"]), 6)
            except ValueError:
                continue
    return dict(sorted(out.items()))


def pink(fname, year, last):
    import openpyxl
    ws = openpyxl.load_workbook(SRC / fname, read_only=True)["Monthly Prices"]
    rows = list(ws.iter_rows(values_only=True))
    hdr = rows[4]
    gi = next(i for i, c in enumerate(hdr) if c and str(c).startswith("Natural gas, Europe"))
    ci = next(i for i, c in enumerate(hdr) if c and str(c).startswith("Coal, South African"))
    out = {}
    for r in rows:
        if r[0] and str(r[0]).startswith(f"{year}M") and int(str(r[0])[5:]) <= last:
            out[str(int(str(r[0])[5:]))] = {"gas_usd_mmbtu": r[gi], "coal_usd_tonne": r[ci]}
    return out


def close(a, b, tol=1e-3):
    if isinstance(a, dict):
        return set(a) == set(b) and all(close(a[k], b[k], tol) for k in a)
    return abs(float(a) - float(b)) <= tol


def main():
    built = {"ttf_daily_frontmonth_acer.json": ttf(),
             "fuel_monthly_2025.json": pink("wb_cmo_monthly_2026-01.xlsx", 2025, 12),
             "fuel_monthly_2026.json": pink("wb_cmo_monthly_2026-09.xlsx", 2026, 8)}
    ok = True
    for name, obj in built.items():
        frozen = json.loads((OUT / name).read_text(encoding="utf-8"))
        same = close(obj, frozen)
        ok &= same
        print(f"{name:34s} rebuilt {len(obj):4d} keys | matches frozen copy: {same}")
        if not same and isinstance(obj, dict):
            diff = [k for k in set(obj) | set(frozen) if k not in obj or k not in frozen
                    or not close(obj[k], frozen[k])]
            print("   differing keys:", sorted(diff)[:10])
        if "--write" in sys.argv:
            (OUT / name).write_text(json.dumps(obj, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
