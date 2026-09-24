"""Hour-level audit of validated GB->NL signs (rule B20), and the rule-robust intersection."""
from __future__ import annotations
import json, pathlib, sys
from collections import Counter
from datetime import datetime, timezone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wedge import inputs_gb_fr_2025_12 as INP                           # noqa: E402
from wedge import run_year_gbfr as RG                                   # noqa: E402
from wedge.run_year import daily_ttf_fuel, obs, techs_for               # noqa: E402
from wedge.mefband import Direction, build_candidates, sign_from_sets   # noqa: E402
from wedge.reclear import units_from, reclear_hour, check_against_mefband  # noqa: E402
from wedge.carbon import load_c26                                   # noqa: E402


def validated_hours(raw, carbon, rule):
    gb_gen, gb_p, nl_gen, nl_p, flow, gb_cap, nl_cap = RG.load_gb_fr(raw, "nl")
    fuel = daily_ttf_fuel(INP.USD_PER_EUR_2025_12)
    out = {}
    hrs = sorted(set(gb_p) & set(nl_p) & set(flow) & set(gb_gen.get("Fossil Gas", {}))
                 & set(nl_gen.get("Fossil gas", {})))
    for h in hrs:
        if flow[h] >= 0:
            continue
        mwh = -flow[h]; gas, coal = fuel(h); pg, pn = carbon(h)
        gt, nt = techs_for(INP.GB_TECHS, gas, coal), techs_for(INP.NL_TECHS, gas, coal)
        cg = build_candidates("GB", obs(gb_gen, gb_cap, gt, h), clearing_price=gb_p[h], p_co2=pg,
                              direction=Direction.REDUCE, price_tolerance=INP.price_tolerance(gb_p[h]),
                              candidate_rule=rule)
        cn = build_candidates("NL", obs(nl_gen, nl_cap, nt, h), clearing_price=nl_p[h], p_co2=pn,
                              direction=Direction.INCREASE, price_tolerance=INP.price_tolerance(nl_p[h]),
                              candidate_rule=rule)
        s, _ = sign_from_sets(cg, cn, INP.ETA_LINK_RANGE_NL)
        if s is None:
            continue
        ux = units_from(gt, {k: v[h] for k, v in gb_gen.items() if h in v}, gb_cap, pg, clearing_price=gb_p[h])
        un = units_from(nt, {k: v[h] for k, v in nl_gen.items() if h in v}, nl_cap, pn, clearing_price=nl_p[h])
        r = reclear_hour(ux, un, flow_mw=mwh, eta=sum(INP.ETA_LINK_RANGE_NL) / 2, cap_flow=1e6)
        c = check_against_mefband(r, s, cg, cn)
        if c["solved"] and c["agreement"] and c["coverage_ok"]:
            out[h] = dict(s=s, mwh=mwh, dE=r.delta_emissions * mwh, gas=gas, pn=pn,
                          gbp=gb_p[h], nlp=nl_p[h],
                          gbset=tuple(t.name for t in cg.members), nlset=tuple(t.name for t in cn.members),
                          nl_coal=nl_gen.get("Fossil hard coal", {}).get(h, 0.0))
    return out


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    if year == 2026:
        C = load_c26()
        carbon = lambda h: (C[str(datetime.fromtimestamp(h, tz=timezone.utc).month)]["gb_dispatch_eur"],
                            C[str(datetime.fromtimestamp(h, tz=timezone.utc).month)]["eua_eur"])
    else:
        carbon = lambda h: (INP.P_CO2_GB_EUR_PER_T, INP.P_CO2_NL_EUR_PER_T)
    raw = pathlib.Path(f"data/raw/gb_fr_{year}")
    a, b = validated_hours(raw, carbon, "marginal_v2"), validated_hours(raw, carbon, "marginal_v3")
    both = {h: a[h] for h in a if h in b and a[h]["s"] == b[h]["s"]}
    mis = [v for v in both.values() if v["s"] > 0]
    print(f"{year}: v2 {len(a)} h, v3 {len(b)} h, INTERSECTION {len(both)} h")
    print(f"  intersection misdirected: {len(mis)} h, {sum(v['mwh'] for v in mis):,.0f} MWh, "
          f"{sum(v['dE'] for v in mis):+,.0f} tCO2")
    print("  candidate-set pairs:", Counter((v["gbset"], v["nlset"]) for v in mis).most_common(4))
    by_m = Counter(datetime.fromtimestamp(h, tz=timezone.utc).strftime("%Y-%m") for h, v in both.items() if v["s"] > 0)
    print("  by month (hours):", dict(sorted(by_m.items())))
    if mis:
        g = sorted(v["gas"] for v in mis); c = sorted(v["nl_coal"] for v in mis)
        print(f"  TTF in those hours: median {g[len(g)//2]:.1f}, min {g[0]:.1f}  | NL coal running: median {c[len(c)//2]:.0f} MW, min {c[0]:.0f} MW")
        print(f"  NL price median {sorted(v['nlp'] for v in mis)[len(mis)//2]:.1f}; GB price median {sorted(v['gbp'] for v in mis)[len(mis)//2]:.1f}")
    pathlib.Path(f"data/processed/gb_nl_validated_intersection_{year}.json").write_text(
        json.dumps({str(h): {k: (list(x) if isinstance(x, tuple) else x) for k, x in v.items()}
                    for h, v in both.items()}, indent=0), encoding="utf-8")
