"""D4: declaration rules compared by the MISDIRECTED LEVY SHARE on tier-B-labelled GB exports.

Any constant positive factor charges every hour, so a binary charge/no-charge accuracy cannot tell
constant rules apart. The informative metric weights each hour by the rule's own factor:

    MisLevyShare(rule) = sum_h e_rule(h) MWh_h 1{sigma=+1}  /  sum_h e_rule(h) MWh_h 1{sigma in +-1}

i.e. the fraction of the rule's total charge that falls on hours where charging RAISES modelled
operational emissions. Rules:
  R-default      0.430 constant (enacted: CO2 / gross FOSSIL generation, Annex IV 1(d))
  R-allsources   0.193 constant (COM(2025) 989 denominator-only; Reg 2025/2621 Annex II)
  R-hourly-avg   GB hourly average intensity: sum(gen_k EF_k) / sum(gen_k)
  R-hourly-marg  GB hourly marginal factor: EF of the exporter-side responder in the dispatch model
                 (the exporter-marginal benchmark; one salient export-side statistic, NOT the best
                 possible export-side rule - the study design)
  R-oracle       charges only hours with sigma = -1 (two-sided; not implementable)
Labels are tier B (model-based oracle). Pooled over NL, BE, FR: a country-of-origin rule cannot see
the destination.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as G, run_year_gbfr as RG     # noqa: E402
from wedge.run_year import daily_ttf_fuel, techs_for                  # noqa: E402
from wedge.reclear import units_from, reclear_hour, robust_sign       # noqa: E402
from wedge.run_tiers import C26, month                                # noqa: E402


def gb_ef_mid():
    return {name: sum(t.ef_el_range) / 2 for name, t in G.GB_TECHS.items()}


def collect(year, mm):
    raw = pathlib.Path(f"data/raw/gb_fr_{year}")
    fuel = daily_ttf_fuel(1.16)
    efm = gb_ef_mid()
    rows = []
    for imp in ("nl", "be", "fr"):
        links, ta, pa, ea = RG.IMPORTERS[imp]
        xg, xp, mg, mp, flow, xc, mc = RG.load_gb_fr(raw, imp)
        for h in sorted(set(xp) & set(mp) & set(flow)):
            d = month(h)
            if d.year != year or d.month > mm or flow[h] >= 0:
                continue
            f = -flow[h]
            gas, coal = fuel(h)
            px, pm = ((G.P_CO2_GB_EUR_PER_T, getattr(G, pa)) if year == 2025 else
                      (C26[str(d.month)]["gb_dispatch_eur"], C26[str(d.month)]["eua_eur"]))
            xt, mt = techs_for(G.GB_TECHS, gas, coal), techs_for(getattr(G, ta), gas, coal)
            gx = {k: v[h] for k, v in xg.items() if h in v}
            gm = {k: v[h] for k, v in mg.items() if h in v}
            if not gx or not gm:
                continue
            sb, _ = robust_sign(lambda q: units_from(xt, gx, xc, px, clearing_price=xp[h], q=q),
                                lambda q: units_from(mt, gm, mc, pm, clearing_price=mp[h], q=q),
                                flow_mw=f, eta_range=getattr(G, ea))
            if sb not in (1, -1):
                continue
            tot = sum(max(v, 0.0) for v in gx.values()) or 1.0
            avg = sum(max(gx.get(k, 0.0), 0.0) * efm.get(k, 0.0) for k in gx) / tot
            r = reclear_hour(units_from(xt, gx, xc, px, clearing_price=xp[h]),
                             units_from(mt, gm, mc, pm, clearing_price=mp[h]),
                             flow_mw=f, eta=sum(getattr(G, ea)) / 2, cap_flow=1e6)
            names = r.exporter_tie or ((r.exporter_responder,) if r.exporter_responder else ())
            ux = {u.name: u.ef_el for u in units_from(xt, gx, xc, px, clearing_price=xp[h])}
            marg = (sum(ux[n] for n in names) / len(names)) if names else avg
            rows.append(dict(mwh=f, sigma=sb, avg=avg, marg=marg))
    return rows


def mis_share(rows, key):
    num = sum(key(r) * r["mwh"] for r in rows if r["sigma"] == 1)
    den = sum(key(r) * r["mwh"] for r in rows)
    return (100 * num / den) if den > 0 else None


def main():
    import wedge.reclear as _RC
    INDEP = "--independent" in sys.argv     # A33 (review B3): same labels as tiers_independent
    _RC.INDEPENDENT_SIGN_CORNERS = INDEP
    out = {}
    rules = {"R-default (0.430)": lambda r: 0.430, "R-allsources (0.193)": lambda r: 0.193,
             "R-hourly-avg": lambda r: r["avg"], "R-hourly-marg": lambda r: r["marg"],
             "R-oracle": lambda r: 1.0 if r["sigma"] == -1 else 0.0}
    for year, mm in ((2025, 12), (2026, 8)):
        rows = collect(year, mm)
        out[str(year)] = {k: mis_share(rows, f) for k, f in rules.items()}
        out[str(year)]["labelled_mwh"] = sum(r["mwh"] for r in rows)
        print(f"{year} (labelled {out[str(year)]['labelled_mwh']:,.0f} MWh) misdirected LEVY share:")
        for k in rules:
            v = out[str(year)][k]
            print(f"   {k:22s} {('n/a' if v is None else f'{v:6.2f} %')}")
    pathlib.Path("data/processed/d4_rules_independent.json" if INDEP else
                 "data/processed/d4_rules.json").write_text(json.dumps(out, indent=1),
                                                            encoding="utf-8")


if __name__ == "__main__":
    main()
