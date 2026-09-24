"""Two-sided sign identification on the frozen border-month RS -> HU, Dec 2025.

🔴 READ THE LABELS. This script computes a **two-sided sign-determinacy rate over charged
import hours**. That is NOT the the study design determinacy rate, which is conditioned on regime
R1, because the regime test needs interconnector NTC / available capacity that is not yet
sourced. The distinction is printed in the output and must survive into any report.

The study design warns explicitly against presenting a one-sided or differently-conditioned
rate as the two-sided volume-weighted determinacy rate. This script therefore reports the
numerator, the denominator and the weighting in full, and names what it is conditioned on.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_rs_hu_2025_12 as INP          # noqa: E402
from wedge.mefband import (                            # noqa: E402
    Direction, Observation, build_candidates, covers, sign_from_sets,
)

RAW = pathlib.Path("data/raw/rs_hu_2025_12")


def load(name: str) -> dict:
    return json.loads((RAW / f"{name}.json").read_text(encoding="utf-8"))


def to_hourly(unix_seconds: list[int], values: list, agg="mean") -> dict[int, float]:
    """Collapse any sub-hourly series onto the hour that contains it.

    Guide rule 2 requires ONE declared harmonisation policy. The coarsest common
    resolution in this month is hourly (RS generation and RS flows are hourly; HU
    generation and both prices are 15-minute), so everything is aggregated UP to the hour.
    Aggregating up never invents data; interpolating down would.
    """
    buckets: dict[int, list[float]] = {}
    for ts, v in zip(unix_seconds, values):
        if v is None:
            continue
        buckets.setdefault((ts // 3600) * 3600, []).append(float(v))
    return {h: sum(vs) / len(vs) for h, vs in buckets.items()}


def series_by_name(doc: dict) -> dict[str, dict[int, float]]:
    ts = doc["unix_seconds"]
    return {pt["name"]: to_hourly(ts, pt["data"]) for pt in doc["production_types"]}


def installed_mw(doc: dict) -> dict[str, float]:
    """Latest yearly installed capacity, GW -> MW."""
    out = {}
    for pt in doc["production_types"]:
        vals = [v for v in pt["data"] if v is not None]
        if vals:
            out[pt["name"]] = float(vals[-1]) * 1000.0
    return out


def build_obs(gen: dict[str, dict[int, float]], cap: dict[str, float],
              techs: dict, hour: int) -> list[Observation]:
    """Observations for one zone-hour.

    🔴 Available capacity is approximated by INSTALLED capacity, because hourly
    availability is an ENTSO-E series we do not have. Installed >= available, so headroom
    is OVERSTATED. That keeps MORE technologies admissible on the 'can increase' side,
    which WIDENS candidate sets and LOWERS determinacy. That is the conservative direction
    and it is stated in the report rather than hidden.
    """
    obs = []
    for name, tech in techs.items():
        if name not in gen:
            continue
        out_mw = gen[name].get(hour)
        if out_mw is None:
            continue
        capacity = cap.get(name, max(out_mw, 0.0))
        obs.append(Observation(tech=tech, output_mw=max(out_mw, 0.0),
                               available_capacity_mw=max(capacity, out_mw),
                               technical_min_mw=0.0))
    return obs


def main(intertemporal_policy: str = "unresolve", candidate_rule: str = "guide_v1") -> int:
    rs_gen = series_by_name(load("rs_generation"))
    hu_gen = series_by_name(load("hu_generation"))
    rs_cap = installed_mw(load("rs_installed"))
    hu_cap = installed_mw(load("hu_installed"))

    pr_rs_doc, pr_hu_doc = load("rs_price"), load("hu_price")
    rs_price = to_hourly(pr_rs_doc["unix_seconds"], pr_rs_doc["price"])
    hu_price = to_hourly(pr_hu_doc["unix_seconds"], pr_hu_doc["price"])

    cbpf = load("rs_cbpf")
    hu_flow = None
    for c in cbpf["countries"]:
        if c.get("name", "").lower().startswith("hungary"):
            # 🔴 UNIT TRAP. The same Energy-Charts API uses THREE different units:
            #   public_power    -> MW   (RS lignite peaks ~3209)
            #   installed_power -> GW   (RS lignite 6.01)
            #   cbpf            -> GW   (RS-HU peaks ~0.7)
            # Treating cbpf as MW understates every flow by 1000x and silently produces a
            # nonsense denominator. Caught by a magnitude sanity check, not by any error.
            hu_flow = {h: v * 1000.0
                       for h, v in to_hourly(cbpf["unix_seconds"], c["data"]).items()}
    if hu_flow is None:
        print("FATAL: no Hungary counterpart series in rs_cbpf")
        return 1

    # magnitude guard: a cross-border flow of a few MW, or of many GW, means the units
    # moved under us again. Fail loudly rather than reporting a wrong denominator.
    peak = max(abs(v) for v in hu_flow.values())
    if not 50.0 <= peak <= 5000.0:
        print(f"FATAL: RS->HU peak flow {peak:.1f} MW is outside the plausible band "
              f"[50, 5000] MW. The upstream units have probably changed. Refusing to run.")
        return 1
    print(f"RS->HU peak flow {peak:.0f} MW (magnitude guard passed)")

    hours = sorted(set(rs_gen.get("Fossil brown coal / lignite", {}))
                   & set(hu_price) & set(rs_price) & set(hu_flow))
    print(f"hours with all inputs present: {len(hours)}")

    # sign convention check
    pos = sum(1 for h in hours if hu_flow[h] > 0)
    neg = sum(1 for h in hours if hu_flow[h] < 0)
    print(f"flow sign: {pos} hours positive, {neg} hours negative "
          f"(Energy-Charts cbpf is signed from the queried country's perspective)")

    tally = Counter()
    mwh_total = mwh_determined = mwh_misdirected = mwh_aligned = 0.0
    reasons = Counter()
    rs_excluded = Counter()
    hu_excluded = Counter()
    examples = []

    for h in hours:
        f = hu_flow[h]
        if f <= 0:                       # not an export from RS into HU
            tally["OUT-OF-SCOPE (no RS->HU import)"] += 1
            continue
        mwh = abs(f)                     # MW over one hour = MWh
        mwh_total += mwh

        c_rs = build_candidates(
            "RS", build_obs(rs_gen, rs_cap, INP.RS_TECHS, h),
            clearing_price=rs_price[h], p_co2=INP.P_CO2_RS_EUR_PER_T,
            direction=Direction.REDUCE,
            price_tolerance=INP.price_tolerance(rs_price[h]),
            candidate_rule=candidate_rule)
        c_hu = build_candidates(
            "HU", build_obs(hu_gen, hu_cap, INP.HU_TECHS, h),
            clearing_price=hu_price[h], p_co2=INP.P_CO2_HU_EUR_PER_T,
            direction=Direction.INCREASE,
            price_tolerance=INP.price_tolerance(hu_price[h]),
            candidate_rule=candidate_rule)

        for n in c_rs.price_excluded:
            rs_excluded[n] += 1
        for n in c_hu.price_excluded:
            hu_excluded[n] += 1

        sigma, reason = sign_from_sets(c_rs, c_hu, INP.ETA_LINK_RANGE,
                                       intertemporal_policy=intertemporal_policy)
        reasons[reason] += 1

        if sigma is None:
            label = ("UNRESOLVED-IDENTIFICATION"
                     if reason in ("empty-candidate-set",
                                   "intertemporal-adjustment-plausible")
                     else "UNDETERMINED-SIGN")
        elif sigma > 0:
            label = "DETERMINED-MISDIRECTED"
            mwh_determined += mwh
            mwh_misdirected += mwh
        else:
            label = "DETERMINED-ALIGNED"
            mwh_determined += mwh
            mwh_aligned += mwh
        tally[label] += 1

        if len(examples) < 3 and sigma is not None:
            examples.append((h, mwh, sigma, [t.name for t in c_rs.members],
                             [t.name for t in c_hu.members]))

    print("\n=== LABEL COUNTS (hours) ===")
    for k, v in tally.most_common():
        print(f"  {k:34s} {v:5d}")

    print("\n=== REASONS for an undecided sign ===")
    for k, v in reasons.most_common():
        print(f"  {k:60s} {v:5d}")

    print("\n=== TWO-SIDED SIGN DETERMINACY, VOLUME WEIGHTED ===")
    print("  🔴 conditioned on: charged RS->HU import hours with all inputs present.")
    print("  🔴 NOT conditioned on regime R1 - no NTC/available-capacity data, so no")
    print("     response test was run. This is therefore NOT the the study design rate.")
    print(f"  numerator   (determined import MWh) = {mwh_determined:,.1f}")
    print(f"  denominator (all import MWh)        = {mwh_total:,.1f}")
    if mwh_total > 0:
        print(f"  determinacy rate                    = {100*mwh_determined/mwh_total:.2f} %")
        print(f"    of which misdirected  (sigma=+1)  = {mwh_misdirected:,.1f} MWh")
        print(f"    of which aligned      (sigma=-1)  = {mwh_aligned:,.1f} MWh")
    else:
        print("  determinacy rate                    = UNDEFINED (empty denominator)")

    print("\n=== price-excluded technologies (coverage diagnostic) ===")
    print("  RS:", dict(rs_excluded) or "none")
    print("  HU:", dict(hu_excluded) or "none")

    print("\n=== worked examples ===")
    for h, mwh, s, rs_m, hu_m in examples:
        print(f"  hour {h}  flow {mwh:.0f} MWh  sigma={s:+d}")
        print(f"     RS candidates: {rs_m}")
        print(f"     HU candidates: {hu_m}")

    out = pathlib.Path(f"data/processed/p05_rs_hu_2025_12_{intertemporal_policy}_{candidate_rule}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "border": "RS->HU", "month": "2025-12",
        "intertemporal_policy": intertemporal_policy, "candidate_rule": candidate_rule,
        "conditioned_on": "charged import hours with all inputs present; NOT regime R1",
        "labels": dict(tally), "reasons": dict(reasons),
        "mwh_total": mwh_total, "mwh_determined": mwh_determined,
        "mwh_misdirected": mwh_misdirected, "mwh_aligned": mwh_aligned,
        "determinacy_rate_pct": (100 * mwh_determined / mwh_total) if mwh_total else None,
        "eta_link_range": list(INP.ETA_LINK_RANGE),
        "p_co2_rs": INP.P_CO2_RS_EUR_PER_T, "p_co2_hu": INP.P_CO2_HU_EUR_PER_T,
        "rs_price_excluded": dict(rs_excluded), "hu_price_excluded": dict(hu_excluded),
    }, indent=2), encoding="utf-8")
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    pol = sys.argv[1] if len(sys.argv) > 1 else "unresolve"
    rule = sys.argv[2] if len(sys.argv) > 2 else "guide_v1"
    print("### intertemporal_policy = " + pol + " | candidate_rule = " + rule)
    sys.exit(main(pol, rule))
