"""Empirical D2 test: does exporter-side information leave a measurable directional gap?

CBAM declares electricity by COUNTRY OF ORIGIN: every GB export carries the same default (0.430)
whatever its destination. So the sharpest evidence for D2 is the SAME GB hour exporting to two
EU zones with OPPOSITE emission directions - no rule that sees only GB-side information can be
right on both. Two label sets, reported separately (never mixed):
  * VALIDATED  - mefband determines AND reclear agrees AND coverage holds (D-014), strict;
  * RECLEAR    - the dispatch model's sign alone, a model-based oracle (secondary, declared).
"""
from __future__ import annotations
import json, pathlib, sys
from collections import defaultdict
from datetime import datetime, timezone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wedge import inputs_gb_fr_2025_12 as INP                           # noqa: E402
from wedge import run_year_gbfr as RG                                   # noqa: E402
from wedge.run_year import daily_ttf_fuel, obs, techs_for               # noqa: E402
from wedge.mefband import Direction, build_candidates, sign_from_sets   # noqa: E402
from wedge.reclear import units_from, reclear_hour, check_against_mefband  # noqa: E402
from wedge.carbon import load_c26                                   # noqa: E402


def labels(raw, imp, year, mm, carbon):
    links, ta, pa, ea = RG.IMPORTERS[imp]
    T, ETA = getattr(INP, ta), getattr(INP, ea)
    gb_gen, gb_p, m_gen, m_p, flow, gb_cap, m_cap = RG.load_gb_fr(raw, imp)
    fuel = daily_ttf_fuel(INP.USD_PER_EUR_2025_12)
    out = {}
    for h in sorted(set(gb_p) & set(m_p) & set(flow) & set(gb_gen.get("Fossil Gas", {}))):
        d = datetime.fromtimestamp(h, tz=timezone.utc)
        if d.year != year or d.month > mm or flow[h] >= 0:
            continue
        mwh = -flow[h]; gas, coal = fuel(h); pg, pm = carbon(h, pa)
        gt, mt = techs_for(INP.GB_TECHS, gas, coal), techs_for(T, gas, coal)
        ux = units_from(gt, {k: v[h] for k, v in gb_gen.items() if h in v}, gb_cap, pg, clearing_price=gb_p[h])
        um = units_from(mt, {k: v[h] for k, v in m_gen.items() if h in v}, m_cap, pm, clearing_price=m_p[h])
        r = reclear_hour(ux, um, flow_mw=mwh, eta=sum(ETA) / 2, cap_flow=1e6)
        if not r.solved:
            continue
        cg = build_candidates("GB", obs(gb_gen, gb_cap, gt, h), clearing_price=gb_p[h], p_co2=pg,
                              direction=Direction.REDUCE, price_tolerance=INP.price_tolerance(gb_p[h]),
                              candidate_rule="marginal_v2")
        cm = build_candidates(imp.upper(), obs(m_gen, m_cap, mt, h), clearing_price=m_p[h], p_co2=pm,
                              direction=Direction.INCREASE, price_tolerance=INP.price_tolerance(m_p[h]),
                              candidate_rule="marginal_v2")
        s, _ = sign_from_sets(cg, cm, ETA)
        c = check_against_mefband(r, s, cg, cm)
        out[h] = dict(mwh=mwh, reclear=r.sign,
                      validated=(s if (c["agreement"] and c["coverage_ok"]) else None))
    return out


if __name__ == "__main__":
    C = load_c26()
    res = {}
    for year, mm in ((2025, 12), (2026, 8)):
        def carbon(h, pa, year=year):
            if year == 2025:
                return INP.P_CO2_GB_EUR_PER_T, getattr(INP, pa)
            m = C[str(datetime.fromtimestamp(h, tz=timezone.utc).month)]
            return m["gb_dispatch_eur"], m["eua_eur"]
        raw = pathlib.Path(f"data/raw/gb_fr_{year}")
        L = {imp: labels(raw, imp, year, mm, carbon) for imp in ("nl", "be", "fr")}
        for key in ("validated", "reclear"):
            by_h = defaultdict(dict)
            for imp, d in L.items():
                for h, v in d.items():
                    if v[key] not in (None, 0):
                        by_h[h][imp] = (v[key], v["mwh"])
            multi = {h: d for h, d in by_h.items() if len(d) >= 2}
            opp = {h: d for h, d in multi.items() if len({s for s, _ in d.values()}) == 2}
            opp_mwh = sum(m for d in opp.values() for _, m in d.values())
            labelled_mwh = sum(m for d in by_h.values() for _, m in d.values())
            # within-border sign variation (the overlap condition needs BOTH signs present)
            per_border = {imp: (sum(v["mwh"] for v in d.values() if v[key] == 1),
                                sum(v["mwh"] for v in d.values() if v[key] == -1)) for imp, d in L.items()}
            res[f"{year}_{key}"] = dict(hours_labelled=len(by_h), multi_dest_hours=len(multi),
                                        opposite_sign_hours=len(opp), opposite_sign_mwh=opp_mwh,
                                        labelled_mwh=labelled_mwh, per_border_mis_ali_mwh=per_border)
            print(f"{year} {key:9s}: labelled GB hours {len(by_h):5d} ({labelled_mwh:>12,.0f} MWh) | "
                  f"multi-destination {len(multi):4d} | OPPOSITE-SIGN same hour {len(opp):4d} ({opp_mwh:>10,.0f} MWh)")
            for imp, (m, a) in per_border.items():
                print(f"      {imp.upper()}: misdirected {m:>12,.0f} MWh | aligned {a:>12,.0f} MWh")
    pathlib.Path("data/processed/d2_test.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
