"""Before/after comparison across candidate rules, both borders, full year 2025.

The study design requires both before and after to be reported whenever a frozen study control
changes. This script runs every rule on every border and writes one table.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_rs_hu_2025_12 as INP          # noqa: E402
from wedge import run_year as RY                        # noqa: E402
from wedge import run_year_gbfr as RG                   # noqa: E402

RULES = sys.argv[1:] or ["marginal_v2", "marginal_v3"]


def row(r: dict, border: str) -> dict:
    v = r["validation"] or {}
    return {
        "border": border, "rule": r["candidate_rule"],
        "determinacy_pct": r["determinacy_pct"],
        "mwh_total": r["mwh_total"],
        "mwh_aligned": r["mwh_aligned"], "mwh_misdirected": r["mwh_misdirected"],
        "agreement_pct": v.get("agreement_pct"),
        "coverage_failure_pct": v.get("coverage_failure_pct"),
        "coverage_failure_volume_pct": v.get("coverage_failure_volume_pct"),
        "responders_outside": v.get("responders_outside"),
        "validated_pct": v.get("validated_determinacy_pct"),
        "val_aligned": v.get("validated_mwh_aligned"),
        "val_misdirected": v.get("validated_mwh_misdirected"),
    }


rows = []
for rule in RULES:
    r1 = RY.run(pathlib.Path("data/raw/rs_hu_2025"),
                RY.daily_ttf_fuel(INP.USD_PER_EUR_2025_12),
                label=f"RS->HU {rule}", candidate_rule=rule)
    rows.append(row(r1, "RS->HU"))
    r2 = RG.run(pathlib.Path("data/raw/gb_fr_2025"), f"GB->FR {rule}", candidate_rule=rule)
    rows.append(row(r2, "GB->FR"))


def f(x, spec=".2f"):
    return "  n/a " if x is None else format(x, spec)


print(f"\n{'border':7s} {'rule':12s} {'determ%':>8s} {'aligned MWh':>12s} {'misdir MWh':>11s} "
      f"{'agree%':>7s} {'covfail%':>9s}  outside-set")
for x in rows:
    print(f"{x['border']:7s} {x['rule']:12s} {f(x['determinacy_pct']):>8s} "
          f"{x['mwh_aligned']:12,.0f} {x['mwh_misdirected']:11,.0f} "
          f"{f(x['agreement_pct']):>7s} {f(x['coverage_failure_pct']):>9s}  "
          f"{x['responders_outside']}")

out = pathlib.Path("data/processed/compare_rules_2025.json")
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"\nwritten -> {out}")
