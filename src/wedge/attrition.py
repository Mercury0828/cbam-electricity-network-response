"""Per-border attrition: how volume shrinks from all imports to rule-robust validated hours.

Zero validated volume is not zero
misdirection, and every share must be read against its full denominator.
"""
from __future__ import annotations
import json, pathlib, sys
from datetime import datetime, timezone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wedge import run_year as RY, run_year_gbfr as RG                   # noqa: E402
from wedge import inputs_gb_fr_2025_12 as G, inputs_rs_hu_2025_12 as R  # noqa: E402
from wedge.carbon import load_c26                                   # noqa: E402


def in_window(h, year, mm):
    d = datetime.fromtimestamp(h, tz=timezone.utc)
    return d.year == year and d.month <= mm


rows = []
C = load_c26()
cm = lambda h: C[str(datetime.fromtimestamp(h, tz=timezone.utc).month)]
for year, mm in ((2025, 12), (2026, 8)):
    gbraw = pathlib.Path(f"data/raw/gb_fr_{year}")
    for imp in ("fr", "nl", "be"):
        carbon = None if year == 2025 else (lambda h: (cm(h)["gb_dispatch_eur"], cm(h)["eua_eur"]))
        _, _, _, _, flow, _, _ = RG.load_gb_fr(gbraw, imp)
        allmwh = sum(-v for h, v in flow.items() if v < 0 and in_window(h, year, mm))
        a = RG.run(gbraw, "", candidate_rule="marginal_v2", imp=imp, carbon=carbon, year=year, max_month=mm)
        b = RG.run(gbraw, "", candidate_rule="marginal_v3", imp=imp, carbon=carbon, year=year, max_month=mm)
        rows.append((f"GB->{imp.upper()}", year, allmwh, a["mwh_total"], a["mwh_determined"],
                     a["validation"]["validated_mwh_misdirected"] + a["validation"]["validated_mwh_aligned"],
                     b["validation"]["validated_mwh_misdirected"] + b["validation"]["validated_mwh_aligned"]))
    rraw = pathlib.Path(f"data/raw/rs_hu_{year}")
    rcarbon = None if year == 2025 else (lambda h: (4.0, cm(h)["eua_eur"]))
    _, _, _, _, flow, _, _ = RY.load_rs_hu(rraw)
    allmwh = sum(v for h, v in flow.items() if v > 0 and in_window(h, year, mm))
    fuel = RY.daily_ttf_fuel(R.USD_PER_EUR_2025_12)
    a = RY.run(rraw, fuel, "", candidate_rule="marginal_v2", carbon=rcarbon, year=year, max_month=mm)
    b = RY.run(rraw, fuel, "", candidate_rule="marginal_v3", carbon=rcarbon, year=year, max_month=mm)
    rows.append(("RS->HU", year, allmwh, a["mwh_total"], a["mwh_determined"],
                 a["validation"]["validated_mwh_misdirected"] + a["validation"]["validated_mwh_aligned"],
                 b["validation"]["validated_mwh_misdirected"] + b["validation"]["validated_mwh_aligned"]))

print(f"{'border':7s} {'year':4s} {'all imports':>13s} {'data-eligible':>14s} {'sign-resolved v2':>17s} {'validated v2':>13s} {'validated v3':>13s}")
for r in rows:
    print(f"{r[0]:7s} {r[1]:4d} {r[2]:13,.0f} {r[3]:14,.0f} {r[4]:17,.0f} {r[5]:13,.0f} {r[6]:13,.0f}")
pathlib.Path("data/processed/attrition.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
