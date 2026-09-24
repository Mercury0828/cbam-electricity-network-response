"""Second border-month run: GB -> FR, December 2025.

Tests the hypothesis that the zero
determinacy on RS -> HU is specific to a mine-mouth-lignite, no-carbon-price border rather
than general to the method. GB and FR are both fuel-priced and both carbon-priced (UK ETS
and EU ETS, which were NOT linked in Dec 2025).

🔴 Same labelling discipline as `run_p05.py`: this is a two-sided, MWh-weighted sign
determinacy over export hours, NOT the the study design R1-conditioned rate.
"""

from __future__ import annotations

import glob
import json
import pathlib
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as INP          # noqa: E402
from wedge.mefband import (                            # noqa: E402
    Direction, Observation, build_candidates, sign_from_sets,
)

RAW = pathlib.Path("data/raw/gb_fr_2025_12")
FR_LINKS = {"France(IFA)", "IFA2 (INTIFA2)", "Eleclink (INTELEC)"}


def hour_of(iso: str) -> int:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    return int(dt.replace(minute=0, second=0, microsecond=0).timestamp())


def mean(d: dict[int, list[float]]) -> dict[int, float]:
    return {k: sum(v) / len(v) for k, v in d.items() if v}


def load_gb_generation() -> dict[str, dict[int, float]]:
    """Elexon AGPT, MW, 30-minute settlement periods -> hourly mean per psrType."""
    acc: dict[str, dict[int, list[float]]] = {}
    for f in sorted(glob.glob(str(RAW / "gb_generation_*.json"))):
        for rec in json.loads(pathlib.Path(f).read_text(encoding="utf-8"))["data"]:
            h = hour_of(rec["startTime"])
            for row in rec["data"]:
                acc.setdefault(row["psrType"], {}).setdefault(h, []).append(
                    float(row["quantity"]))
    return {k: mean(v) for k, v in acc.items()}


def load_gb_price() -> dict[int, float]:
    """Elexon MID, converted GBP -> EUR.

    🔴 CURRENCY TRAP. Elexon publishes market index prices in GBP/MWh. Every SRMC band in
    `inputs_gb_fr_2025_12` is denominated in EUR/MWh (EUR fuel cost, EUR carbon price).
    Comparing a GBP price against a EUR band silently shifts the whole GB merit order by the
    exchange rate and changes which technologies survive the price filter. Converted here,
    once, using the Bank of England December 2025 monthly average.
    """
    acc: dict[int, list[float]] = {}
    for f in sorted(glob.glob(str(RAW / "gb_price_*.json"))):
        for rec in json.loads(pathlib.Path(f).read_text(encoding="utf-8"))["data"]:
            if rec.get("dataProvider") != "APXMIDP":
                continue
            if rec.get("volume", 0) <= 0:
                continue
            acc.setdefault(hour_of(rec["startTime"]), []).append(
                float(rec["price"]) * INP.GBP_PER_EUR_2025_12)
    return mean(acc)


def load_gb_to_fr_flow() -> dict[int, float]:
    """INTOUTHH, MW. Positive = import INTO GB, so GB->FR export is the NEGATIVE side."""
    acc: dict[int, list[float]] = {}
    per_hour: dict[int, dict[str, float]] = {}
    for f in sorted(glob.glob(str(RAW / "gb_interconnector_*.json"))):
        for rec in json.loads(pathlib.Path(f).read_text(encoding="utf-8"))["data"]:
            if rec.get("interconnectorName") not in FR_LINKS:
                continue
            h = hour_of(rec["startTime"])
            per_hour.setdefault(h, {}).setdefault(rec["interconnectorName"], 0.0)
            per_hour[h][rec["interconnectorName"]] += float(rec["generation"])
    # two settlement periods per hour per link -> average, then sum across the three links
    for h, links in per_hour.items():
        acc[h] = [sum(v / 2.0 for v in links.values())]
    return mean(acc)


def load_fr(name: str) -> dict:
    return json.loads((RAW / f"{name}.json").read_text(encoding="utf-8"))


def to_hourly(unix_seconds, values) -> dict[int, float]:
    b: dict[int, list[float]] = {}
    for ts, v in zip(unix_seconds, values):
        if v is not None:
            b.setdefault((ts // 3600) * 3600, []).append(float(v))
    return mean(b)


def build_obs(gen, cap, techs, hour):
    obs = []
    for name, tech in techs.items():
        if name not in gen:
            continue
        out = gen[name].get(hour)
        if out is None:
            continue
        c = cap.get(name, max(out, 0.0))
        obs.append(Observation(tech, max(out, 0.0), max(c, out), 0.0))
    return obs


def main(policy: str = "unresolve", candidate_rule: str = "guide_v1") -> int:
    gb_gen = load_gb_generation()
    gb_price = load_gb_price()
    flow = load_gb_to_fr_flow()

    frd = load_fr("fr_generation")
    fr_gen = {pt["name"]: to_hourly(frd["unix_seconds"], pt["data"])
              for pt in frd["production_types"]}
    frp = load_fr("fr_price")
    fr_price = to_hourly(frp["unix_seconds"], frp["price"])

    # capacity proxy: observed peak over the month per technology (no installed series here).
    gb_cap = {k: max(v.values()) * 1.15 for k, v in gb_gen.items() if v}
    fr_cap = {k: max(v.values()) * 1.15 for k, v in fr_gen.items() if v}

    hours = sorted(set(gb_price) & set(fr_price) & set(flow)
                   & set(gb_gen.get("Fossil Gas", {})) & set(fr_gen.get("Nuclear", {})))
    print(f"hours with all inputs present: {len(hours)}")
    exp_h = [h for h in hours if flow[h] < 0]
    print(f"GB->FR export hours (flow negative): {len(exp_h)}")
    if exp_h:
        print(f"peak GB->FR export: {max(-flow[h] for h in exp_h):.0f} MW")

    tally, reasons = Counter(), Counter()
    gb_x, fr_x = Counter(), Counter()
    tot = det = mis = ali = 0.0
    examples = []

    for h in hours:
        if flow[h] >= 0:
            tally["OUT-OF-SCOPE (no GB->FR export)"] += 1
            continue
        mwh = -flow[h]
        tot += mwh
        c_gb = build_candidates("GB", build_obs(gb_gen, gb_cap, INP.GB_TECHS, h),
                                clearing_price=gb_price[h], p_co2=INP.P_CO2_GB_EUR_PER_T,
                                direction=Direction.REDUCE,
                                price_tolerance=INP.price_tolerance(gb_price[h]),
                                candidate_rule=candidate_rule)
        c_fr = build_candidates("FR", build_obs(fr_gen, fr_cap, INP.FR_TECHS, h),
                                clearing_price=fr_price[h], p_co2=INP.P_CO2_FR_EUR_PER_T,
                                direction=Direction.INCREASE,
                                price_tolerance=INP.price_tolerance(fr_price[h]),
                                candidate_rule=candidate_rule)
        for n in c_gb.price_excluded:
            gb_x[n] += 1
        for n in c_fr.price_excluded:
            fr_x[n] += 1

        sigma, reason = sign_from_sets(c_gb, c_fr, INP.ETA_LINK_RANGE,
                                       intertemporal_policy=policy)
        reasons[reason] += 1
        if sigma is None:
            tally["UNRESOLVED-IDENTIFICATION" if reason in (
                "empty-candidate-set", "intertemporal-adjustment-plausible")
                else "UNDETERMINED-SIGN"] += 1
        else:
            det += mwh
            if sigma > 0:
                mis += mwh
                tally["DETERMINED-MISDIRECTED"] += 1
            else:
                ali += mwh
                tally["DETERMINED-ALIGNED"] += 1
            if len(examples) < 4:
                examples.append((h, mwh, sigma,
                                 [t.name for t in c_gb.members],
                                 [t.name for t in c_fr.members]))

    print("\n=== LABELS (hours) ===")
    for k, v in tally.most_common():
        print(f"  {k:36s} {v:5d}")
    print("\n=== reasons undecided ===")
    for k, v in reasons.most_common():
        print(f"  {k:62s} {v:5d}")

    print("\n=== TWO-SIDED SIGN DETERMINACY, VOLUME WEIGHTED ===")
    print("  conditioned on: GB->FR export hours with all inputs present. NOT regime R1.")
    print(f"  numerator   (determined MWh) = {det:,.1f}")
    print(f"  denominator (all export MWh) = {tot:,.1f}")
    if tot > 0:
        print(f"  determinacy rate             = {100*det/tot:.2f} %")
        print(f"    misdirected (sigma=+1)     = {mis:,.1f} MWh  ({100*mis/tot:.2f} %)")
        print(f"    aligned     (sigma=-1)     = {ali:,.1f} MWh  ({100*ali/tot:.2f} %)")
    else:
        print("  determinacy rate             = UNDEFINED (empty denominator)")

    print("\n=== price-excluded (coverage diagnostic) ===")
    print("  GB:", dict(gb_x) or "none")
    print("  FR:", dict(fr_x) or "none")
    print("\n=== examples ===")
    for h, mwh, s, g, f in examples:
        print(f"  h={h} {mwh:7.0f} MWh sigma={s:+d}\n     GB: {g}\n     FR: {f}")

    out = pathlib.Path(f"data/processed/p05_gb_fr_2025_12_{policy}_{candidate_rule}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "border": "GB->FR", "month": "2025-12", "intertemporal_policy": policy, "candidate_rule": candidate_rule,
        "conditioned_on": "GB->FR export hours, all inputs present; NOT regime R1",
        "labels": dict(tally), "reasons": dict(reasons),
        "mwh_total": tot, "mwh_determined": det,
        "mwh_misdirected": mis, "mwh_aligned": ali,
        "determinacy_rate_pct": (100 * det / tot) if tot else None,
        "eta_link_range": list(INP.ETA_LINK_RANGE),
        "p_co2_gb": INP.P_CO2_GB_EUR_PER_T, "p_co2_fr": INP.P_CO2_FR_EUR_PER_T,
        "gb_price_excluded": dict(gb_x), "fr_price_excluded": dict(fr_x),
    }, indent=2), encoding="utf-8")
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    pol = sys.argv[1] if len(sys.argv) > 1 else "unresolve"
    rule = sys.argv[2] if len(sys.argv) > 2 else "guide_v1"
    print("### intertemporal_policy = " + pol + " | candidate_rule = " + rule)
    sys.exit(main(pol, rule))
