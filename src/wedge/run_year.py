"""Full-year identification + re-clearing validation for RS -> HU.

Generalises the December-2025 de-risk run to a full calendar year, with two improvements
that are REMEDIATIONS of diagnosed defects rather than tuning:

1. **Fuel prices vary by hour's month** instead of one December value (barrier B21). December
   2025 was the cheapest gas month of the year, which is part of why it looked so bad.
2. **Fuel is injected as a function** `fuel_for_hour(h) -> (gas_eur_mwh_th, coal_eur_mwh_th)`,
   so a daily series can replace the monthly one without touching the pipeline.

🔴 The identification rule, the tolerance, the loss factor and the emission factors are the
frozen ones (D-009, D-013). Nothing here is tuned against a determinacy rate.

🔴 Same labelling discipline as before: this is two-sided, MWh-weighted sign determinacy over
charged import hours, NOT the the study design R1-conditioned rate (no NTC, so no response test).
"""

from __future__ import annotations

import dataclasses
import glob
import json
import pathlib
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_rs_hu_2025_12 as INP          # noqa: E402
from wedge.mefband import (                            # noqa: E402
    Direction, Observation, Tech, build_candidates, sign_from_sets,
)
from wedge.reclear import check_against_mefband, reclear_hour, units_from  # noqa: E402

MMBTU_PER_MWH = 3.412142


# --------------------------------------------------------------------- loading
def _merge(pattern: str, key: str = "production_types") -> dict:
    """Concatenate monthly Energy-Charts chunks into one hourly series per name."""
    series: dict[str, dict[int, list[float]]] = {}
    for f in sorted(glob.glob(pattern)):
        d = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        ts = d["unix_seconds"]
        if key == "production_types":
            items = [(pt["name"], pt["data"]) for pt in d["production_types"]]
        elif key == "price":
            items = [("price", d["price"])]
        else:                                      # cbpf
            items = [(c["name"], c["data"]) for c in d["countries"]]
        for name, vals in items:
            b = series.setdefault(name, {})
            for t, v in zip(ts, vals):
                if v is not None:
                    b.setdefault((t // 3600) * 3600, []).append(float(v))
    return {n: {h: sum(v) / len(v) for h, v in b.items()} for n, b in series.items()}


def load_rs_hu(raw: pathlib.Path):
    rs_gen = _merge(str(raw / "rs_generation_*.json"))
    hu_gen = _merge(str(raw / "hu_generation_*.json"))
    rs_price = _merge(str(raw / "rs_price_*.json"), "price")["price"]
    hu_price = _merge(str(raw / "hu_price_*.json"), "price")["price"]
    cb = _merge(str(raw / "rs_cbpf_*.json"), "cbpf")
    hu_name = next(n for n in cb if n.lower().startswith("hungary"))
    flow = {h: v * 1000.0 for h, v in cb[hu_name].items()}     # 🔴 cbpf is GW (B17)

    def installed(fn):
        d = json.loads((raw / fn).read_text(encoding="utf-8"))
        return {pt["name"]: float([v for v in pt["data"] if v is not None][-1]) * 1000.0
                for pt in d["production_types"] if any(v is not None for v in pt["data"])}
    return rs_gen, hu_gen, rs_price, hu_price, flow, installed("rs_installed.json"), \
        installed("hu_installed.json")


# --------------------------------------------------------------------- fuel
def monthly_fuel(usd_per_eur: float) -> Callable[[int], tuple[float, float]]:
    """World Bank Pink Sheet, by the hour's calendar YEAR and month (UTC).

    Reads `fuel_monthly_<year>.json` for whichever year the hour falls in, so the same
    pipeline runs the retrospective year (2025) and the charged period (2026). A missing
    year raises rather than silently borrowing another year's prices.
    """
    tabs: dict[int, dict] = {}

    def tab_for(y: int) -> dict:
        if y not in tabs:
            fp = pathlib.Path(f"data/processed/fuel_monthly_{y}.json")
            if not fp.exists():
                raise KeyError(f"no World Bank fuel table for {y}")
            tabs[y] = json.loads(fp.read_text(encoding="utf-8"))
        return tabs[y]

    def fx(d) -> float:
        # a sourced month-specific ECB rate where one exists (2026), else the declared value
        cp = pathlib.Path(f"data/processed/carbon_monthly_{d.year}.json")
        if cp.exists():
            row = json.loads(cp.read_text(encoding="utf-8")).get(str(d.month))
            if row and "usd_per_eur" in row:
                return row["usd_per_eur"]
        return usd_per_eur

    def f(h: int) -> tuple[float, float]:
        d = datetime.fromtimestamp(h, tz=timezone.utc)
        r = tab_for(d.year)[str(d.month)]
        gas = r["gas_usd_mmbtu"] * MMBTU_PER_MWH / fx(d)
        coal = (r["coal_usd_tonne"] / 6.978) / fx(d)
        return gas, coal
    return f


def daily_ttf_fuel(usd_per_eur: float) -> Callable[[int], tuple[float, float]]:
    """Daily TTF front-month (ACER-derived, B25) for gas; World Bank monthly for coal.

    ACER publishes on working days only. For a non-trading day the most recent PRIOR
    published value is used - carried FORWARD only, never backward, because a delivery-day
    fuel cost cannot depend on a price published afterwards. No value is invented; an
    existing observation is reused.
    It is a FRONT-MONTH FUTURES settlement, not a day-ahead price (B25), and is named so.
    """
    import bisect
    ttf = json.loads(pathlib.Path("data/processed/ttf_daily_frontmonth_acer.json")
                     .read_text(encoding="utf-8"))
    dates = sorted(ttf)
    coal_m = monthly_fuel(usd_per_eur)

    def f(h: int) -> tuple[float, float]:
        d = datetime.fromtimestamp(h, tz=timezone.utc).strftime("%Y-%m-%d")
        i = bisect.bisect_right(dates, d) - 1
        if i < 0:
            raise KeyError(f"no ACER TTF observation on or before {d}")
        return ttf[dates[i]], coal_m(h)[1]
    return f


def techs_for(base: dict[str, Tech], gas: float, coal: float | None = None) -> dict[str, Tech]:
    """Re-price gas-fired (and, if given, hard-coal) technologies for this hour's fuel prices.

    🔴 A29: the monthly coal price was computed by the fuel lookup and then
    discarded, so hard coal stayed at its December-2025 cost in every hour of both years. Coal is
    now re-priced too. Lignite is cost-unidentified (mine-mouth) and is never re-priced.
    """
    out = {}
    for k, t in base.items():
        kl = k.lower()
        if t.fuel_th is not None and not t.cost_unidentified and "gas" in kl:
            out[k] = dataclasses.replace(t, fuel_th=gas)
        elif (coal is not None and t.fuel_th is not None and not t.cost_unidentified
              and "hard coal" in kl):
            out[k] = dataclasses.replace(t, fuel_th=coal)
        else:
            out[k] = t
    return out


def obs(gen, cap, techs, h):
    o = []
    for name, tech in techs.items():
        if name in gen and h in gen[name]:
            out = max(gen[name][h], 0.0)
            o.append(Observation(tech, out, max(cap.get(name, out), out), 0.0))
    return o


# --------------------------------------------------------------------- run
def run(raw: pathlib.Path, fuel: Callable[[int], tuple[float, float]], label: str,
        candidate_rule: str = "marginal_v2", policy: str = "unresolve",
        validate: bool = True, carbon=None, year: int = 2025, max_month: int = 12) -> dict:
    """`carbon(h) -> (p_rs, p_hu)`, EUR/tCO2. Default: the 2025 constants (RS 0, HU Dec-2025 EUA)."""
    if carbon is None:
        carbon = lambda h: (INP.P_CO2_RS_EUR_PER_T, INP.P_CO2_HU_EUR_PER_T)   # noqa: E731
    rs_gen, hu_gen, rs_p, hu_p, flow, rs_cap, hu_cap = load_rs_hu(raw)

    peak = max(abs(v) for v in flow.values())
    assert 50.0 <= peak <= 5000.0, f"flow magnitude guard failed: peak {peak:.1f} MW"

    hours = sorted(set(rs_gen.get("Fossil brown coal / lignite", {}))
                   & set(rs_p) & set(hu_p) & set(flow))
    # Energy-Charts month boundaries are LOCAL time, so a UTC-hour series spills an hour or two
    # into the neighbouring year. Keep only hours inside the study year in UTC (A18).
    hours = [h for h in hours if datetime.fromtimestamp(h, tz=timezone.utc).year == year]
    # charged window is bounded by the fuel/carbon series coverage (Jan-Aug 2026)
    hours = [h for h in hours if datetime.fromtimestamp(h, tz=timezone.utc).month <= max_month]
    imp = [h for h in hours if flow[h] > 0]
    eta_mid = sum(INP.ETA_LINK_RANGE) / 2.0

    tally, cov_miss = Counter(), Counter()
    tot = det = mis = ali = 0.0
    agree_h = cov_fail_h = det_h = 0
    agree_mwh = cov_fail_mwh = 0.0
    val_ali = val_mis = 0.0
    by_month = {}

    for h in imp:
        mwh = flow[h]
        tot += mwh
        gas, coal = fuel(h)
        pc_rs, pc_hu = carbon(h)
        rs_t = techs_for(INP.RS_TECHS, gas, coal)
        hu_t = techs_for(INP.HU_TECHS, gas, coal)

        c_rs = build_candidates("RS", obs(rs_gen, rs_cap, rs_t, h),
                                clearing_price=rs_p[h], p_co2=pc_rs,
                                direction=Direction.REDUCE,
                                price_tolerance=INP.price_tolerance(rs_p[h]),
                                candidate_rule=candidate_rule)
        c_hu = build_candidates("HU", obs(hu_gen, hu_cap, hu_t, h),
                                clearing_price=hu_p[h], p_co2=pc_hu,
                                direction=Direction.INCREASE,
                                price_tolerance=INP.price_tolerance(hu_p[h]),
                                candidate_rule=candidate_rule)
        sigma, reason = sign_from_sets(c_rs, c_hu, INP.ETA_LINK_RANGE,
                                       intertemporal_policy=policy)

        m = datetime.fromtimestamp(h, tz=timezone.utc).month
        bm = by_month.setdefault(m, [0.0, 0.0])
        bm[0] += mwh

        if sigma is None:
            tally["UNDETERMINED" if reason == "emission intervals overlap"
                  else "UNRESOLVED"] += 1
        else:
            det += mwh
            det_h += 1
            bm[1] += mwh
            if sigma > 0:
                mis += mwh
                tally["DETERMINED-MISDIRECTED"] += 1
            else:
                ali += mwh
                tally["DETERMINED-ALIGNED"] += 1

            if validate:
                ux = units_from(rs_t, {k: v[h] for k, v in rs_gen.items() if h in v},
                                rs_cap, pc_rs, clearing_price=rs_p[h])
                um = units_from(hu_t, {k: v[h] for k, v in hu_gen.items() if h in v},
                                hu_cap, pc_hu, clearing_price=hu_p[h])
                res = reclear_hour(ux, um, flow_mw=mwh, eta=eta_mid, cap_flow=1e6)
                chk = check_against_mefband(res, sigma, c_rs, c_hu)
                if chk["solved"]:
                    if chk["agreement"]:
                        agree_h += 1
                        agree_mwh += mwh
                    if chk["agreement"] and chk["coverage_ok"]:
                        if sigma > 0:
                            val_mis += mwh
                        else:
                            val_ali += mwh
                    if not chk["coverage_ok"]:
                        cov_fail_h += 1
                        cov_fail_mwh += mwh
                        for x in chk["responders_outside_candidate_set"]:
                            cov_miss[x] += 1

    res = {
        "label": label, "border": "RS->HU", "year": year,
        "candidate_rule": candidate_rule, "intertemporal_policy": policy,
        "conditioned_on": "charged RS->HU import hours, all inputs present; NOT regime R1",
        "import_hours": len(imp), "mwh_total": tot, "mwh_determined": det,
        "mwh_misdirected": mis, "mwh_aligned": ali,
        "determinacy_pct": 100 * det / tot if tot else None,
        "labels": dict(tally),
        "validation": {
            "determined_hours": det_h,
            "agreement_pct": 100 * agree_h / det_h if det_h else None,
            "agreement_volume_pct": 100 * agree_mwh / det if det else None,
            "coverage_failure_pct": 100 * cov_fail_h / det_h if det_h else None,
            "coverage_failure_volume_pct": 100 * cov_fail_mwh / det if det else None,
            "responders_outside": dict(cov_miss),
            "validated_mwh_aligned": val_ali,
            "validated_mwh_misdirected": val_mis,
            "validated_determinacy_pct": 100 * (val_ali + val_mis) / tot if tot else None,
        } if validate else None,
        "determinacy_by_month_pct": {m: (100 * v[1] / v[0] if v[0] else None)
                                     for m, v in sorted(by_month.items())},
    }
    return res


def report(r: dict) -> None:
    print(f"\n######## {r['label']} ########")
    print(f"  import hours {r['import_hours']}, total {r['mwh_total']:,.0f} MWh")
    print(f"  determinacy        = {r['determinacy_pct']:.2f} %  "
          f"({r['mwh_determined']:,.0f} MWh)")
    print(f"    aligned  (s=-1)  = {r['mwh_aligned']:,.0f} MWh")
    print(f"    misdir.  (s=+1)  = {r['mwh_misdirected']:,.0f} MWh")
    v = r["validation"]
    if v and v["determined_hours"]:
        print(f"  AGREEMENT          = {v['agreement_pct']:.2f} %  "
              f"(vol {v['agreement_volume_pct']:.2f} %)")
        print(f"  COVERAGE FAILURE   = {v['coverage_failure_pct']:.2f} %  "
              f"(vol {v['coverage_failure_volume_pct']:.2f} %)")
        if v["responders_outside"]:
            print(f"  outside-set        = {v['responders_outside']}")
    print("  determinacy by month:",
          {m: (round(x, 1) if x is not None else None)
           for m, x in r["determinacy_by_month_pct"].items()})


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "daily"
    fx = INP.USD_PER_EUR_2025_12
    fuel = daily_ttf_fuel(fx) if mode == "daily" else monthly_fuel(fx)
    r = run(pathlib.Path("data/raw/rs_hu_2025"), fuel, label=f"RS->HU 2025, {mode} fuel")
    report(r)
    out = pathlib.Path(f"data/processed/year_rs_hu_2025_{mode}fuel.json")
    out.write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(f"\nwritten -> {out}")
