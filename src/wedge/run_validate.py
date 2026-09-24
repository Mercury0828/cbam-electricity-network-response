"""Validate `mefband` signs against the re-clearing cross-check.

Runs over the hours of a border-month and reports, SEPARATELY:

  * **agreement rate** — of hours where `mefband` determined a sign, how often the
    re-clearing model's emission response has the same sign;
  * 🔴 **coverage failure rate** — of those hours, how often the unit that actually responded
    in the model lies OUTSIDE `mefband`'s candidate set. A coverage failure invalidates the
    determined label whatever the agreement rate says (kill criterion K5b).

Both numbers are reported for the determined hours AND for the undetermined ones, because a
coverage failure in an hour `mefband` refused to decide is still information about the filter.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_rs_hu_2025_12 as INP          # noqa: E402
from wedge import run_p05 as R                          # noqa: E402
from wedge.mefband import Direction, build_candidates, sign_from_sets   # noqa: E402
from wedge.reclear import check_against_mefband, reclear_hour, units_from  # noqa: E402


def main(candidate_rule: str = "marginal_v2", policy: str = "unresolve",
         max_hours: int | None = None) -> int:
    rs_gen = R.series_by_name(R.load("rs_generation"))
    hu_gen = R.series_by_name(R.load("hu_generation"))
    rs_cap = R.installed_mw(R.load("rs_installed"))
    hu_cap = R.installed_mw(R.load("hu_installed"))
    prs, phu = R.load("rs_price"), R.load("hu_price")
    rs_price = R.to_hourly(prs["unix_seconds"], prs["price"])
    hu_price = R.to_hourly(phu["unix_seconds"], phu["price"])
    cb = R.load("rs_cbpf")
    flow = {}
    for c in cb["countries"]:
        if c.get("name", "").lower().startswith("hungary"):
            flow = {h: v * 1000.0 for h, v in
                    R.to_hourly(cb["unix_seconds"], c["data"]).items()}

    hours = sorted(set(rs_gen.get("Fossil brown coal / lignite", {}))
                   & set(hu_price) & set(rs_price) & set(flow))
    imp = [h for h in hours if flow[h] > 0]
    if max_hours:
        imp = imp[:max_hours]
    print(f"validating {len(imp)} RS->HU import hours "
          f"(candidate_rule={candidate_rule}, intertemporal={policy})")

    eta_mid = sum(INP.ETA_LINK_RANGE) / 2.0
    stats = Counter()
    misses = Counter()
    det_mwh = agree_mwh = covfail_mwh = 0.0

    for h in imp:
        c_rs = build_candidates("RS", R.build_obs(rs_gen, rs_cap, INP.RS_TECHS, h),
                                clearing_price=rs_price[h], p_co2=INP.P_CO2_RS_EUR_PER_T,
                                direction=Direction.REDUCE,
                                price_tolerance=INP.price_tolerance(rs_price[h]),
                                candidate_rule=candidate_rule)
        c_hu = build_candidates("HU", R.build_obs(hu_gen, hu_cap, INP.HU_TECHS, h),
                                clearing_price=hu_price[h], p_co2=INP.P_CO2_HU_EUR_PER_T,
                                direction=Direction.INCREASE,
                                price_tolerance=INP.price_tolerance(hu_price[h]),
                                candidate_rule=candidate_rule)
        sigma, _ = sign_from_sets(c_rs, c_hu, INP.ETA_LINK_RANGE,
                                  intertemporal_policy=policy)

        ux = units_from(INP.RS_TECHS, {k: v[h] for k, v in rs_gen.items() if h in v},
                        rs_cap, INP.P_CO2_RS_EUR_PER_T)
        um = units_from(INP.HU_TECHS, {k: v[h] for k, v in hu_gen.items() if h in v},
                        hu_cap, INP.P_CO2_HU_EUR_PER_T)
        res = reclear_hour(ux, um, flow_mw=flow[h], eta=eta_mid, cap_flow=1e6)
        chk = check_against_mefband(res, sigma, c_rs, c_hu)

        if not chk["solved"]:
            stats["lp-unsolved"] += 1
            continue
        determined = sigma is not None
        stats["determined" if determined else "undetermined"] += 1
        if determined:
            det_mwh += flow[h]
            if chk["agreement"]:
                stats["agree"] += 1
                agree_mwh += flow[h]
            else:
                stats["disagree"] += 1
            if not chk["coverage_ok"]:
                stats["coverage-fail-determined"] += 1
                covfail_mwh += flow[h]
                for m in chk["responders_outside_candidate_set"]:
                    misses[m] += 1
        else:
            if not chk["coverage_ok"]:
                stats["coverage-fail-undetermined"] += 1
                for m in chk["responders_outside_candidate_set"]:
                    misses[m] += 1

    print("\n=== counts ===")
    for k, v in stats.most_common():
        print(f"  {k:28s} {v:5d}")

    det = stats["determined"]
    print("\n=== (1) AGREEMENT, on determined hours ===")
    if det:
        print(f"  agreement rate      = {100*stats['agree']/det:.2f} %  "
              f"({stats['agree']}/{det} hours)")
        print(f"  volume weighted     = {100*agree_mwh/det_mwh:.2f} %  "
              f"({agree_mwh:,.0f} / {det_mwh:,.0f} MWh)")
    else:
        print("  UNDEFINED - no determined hours")

    print("\n=== (2) 🔴 COVERAGE, reported separately (K5b) ===")
    if det:
        cf = stats["coverage-fail-determined"]
        print(f"  coverage FAILURE rate on determined hours = {100*cf/det:.2f} %  "
              f"({cf}/{det})")
        print(f"  volume weighted                           = "
              f"{100*covfail_mwh/det_mwh:.2f} %  ({covfail_mwh:,.0f} MWh)")
        if cf:
            print("  🔴 A DEMONSTRATED COVERAGE FAILURE ON DETERMINED HOURS IS K5b.")
    else:
        print("  UNDEFINED - no determined hours")
    print(f"  coverage failures on UNDETERMINED hours   = "
          f"{stats['coverage-fail-undetermined']}")

    print("\n=== responders found outside the candidate set ===")
    for k, v in misses.most_common(12):
        print(f"  {k:34s} {v}")

    out = pathlib.Path(
        f"data/processed/validate_rs_hu_2025_12_{policy}_{candidate_rule}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "border": "RS->HU", "month": "2025-12",
        "candidate_rule": candidate_rule, "intertemporal_policy": policy,
        "counts": dict(stats),
        "agreement_rate_pct": (100*stats["agree"]/det) if det else None,
        "agreement_rate_volume_pct": (100*agree_mwh/det_mwh) if det_mwh else None,
        "coverage_failure_rate_determined_pct": (
            100*stats["coverage-fail-determined"]/det) if det else None,
        "coverage_failure_volume_pct": (100*covfail_mwh/det_mwh) if det_mwh else None,
        "responders_outside_candidate_set": dict(misses),
    }, indent=2), encoding="utf-8")
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    rule = sys.argv[1] if len(sys.argv) > 1 else "marginal_v2"
    pol = sys.argv[2] if len(sys.argv) > 2 else "unresolve"
    sys.exit(main(rule, pol))
