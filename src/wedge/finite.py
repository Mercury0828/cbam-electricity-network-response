"""Finite-charge flow response: D1 consequences and D4.

Per hour, a two-zone economic dispatch where the border flow f is a DECISION variable and the
charge enters the objective:

    min_f  C_x(D_x + f/eta) + C_m(D_m - f) + c*f,      0 <= f <= cap

C_x and C_m are merit-order cost curves from the same technology stacks as `reclear` (midpoints),
with must-run fixed, variable renewables capped at output, offline thermal held (A10, A12, A16,
A23). Demand is held fixed (A3). The two-zone boundary ignores re-routing via third zones
(declared limitation).

Scenarios (charge per delivered MWh):
  none         c = 0
  enacted      c = P_cert x declared default (GB 0.430, RS 1.041), GROSS. net = 0 equals 'none',
               so the pair (none, enacted) brackets the unpublished Art. 9 deduction.
  all_sources  c = P_cert x the country's grid factor (GB 0.193, RS 0.729; Reg 2025/2621
               Annex II) - the DENOMINATOR-ONLY experiment of COM(2025) 989, not the whole
               proposed reform package.

Effects are MODEL differences between scenarios (delta method). The model is not asked to
reproduce observed levels, but its fit to the observed flow under the enacted charge is reported
as the calibration check, and it is part of the result.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as G, inputs_rs_hu_2025_12 as R      # noqa: E402
from wedge import run_year as RY, run_year_gbfr as RG                        # noqa: E402
from wedge.run_year import daily_ttf_fuel, techs_for                         # noqa: E402
from wedge.reclear import units_from                                         # noqa: E402
from wedge.run_tiers import C26, month                                       # noqa: E402
from wedge.regime import CAP                                                 # noqa: E402

CERT = {1: 75.36, 2: 75.28}   # EUR/certificate, 2026 Q1/Q2 (law F7)
STEP = 10.0                   # MW grid for the flow


def curve(units):
    floor = sum(u.min_mw for u in units)
    base_cost = sum(u.min_mw * u.srmc for u in units)
    base_em = sum(u.min_mw * u.ef_el for u in units)
    segs = sorted(((u.capacity_mw - u.min_mw, u.srmc, u.ef_el) for u in units
                   if u.capacity_mw > u.min_mw + 1e-9), key=lambda s: s[1])
    return floor, base_cost, base_em, segs


def cost_em(cv, q):
    floor, c, e, segs = cv
    if q < floor - 1e-6:
        return None
    rest = q - floor
    for w, p, ef in segs:
        t = min(rest, w)
        c += t * p
        e += t * ef
        rest -= t
        if rest <= 1e-9:
            return c, e
    return None if rest > 1e-6 else (c, e)


def solve(cx, cm, dx, dm, eta, cap, charge):
    best = None
    f = 0.0
    while f <= cap + 1e-9:
        a, b = cost_em(cx, dx + f / eta), cost_em(cm, dm - f)
        if a is not None and b is not None:
            tot = a[0] + b[0] + charge * f
            if best is None or tot < best[0] - 1e-9:
                best = (tot, f, a[1] + b[1])
        f += STEP
    return best


def run(border, loader, X, M, carbon_fn, eta_rng, e_dec, e_grid, sign, cap_fn, mm=8):
    xg, xp, mg, mp, flow, xc, mc = loader()
    fuel = daily_ttf_fuel(1.16)
    eta = sum(eta_rng) / 2
    agg = {k: dict(flow_mwh=0.0, em_t=0.0, charge_eur=0.0)
           for k in ("none", "enacted", "all_sources")}
    obs_f, mod_f = [], []
    for h in sorted(set(xp) & set(mp) & set(flow)):
        d = month(h)
        if d.year != 2026 or d.month > mm:
            continue
        q = (d.month - 1) // 3 + 1
        if q not in CERT:
            continue
        cap = cap_fn(h)
        if cap is None:
            continue
        fo = max(sign * flow[h], 0.0)
        gas, coal = fuel(h)
        px, pm = carbon_fn(h)
        xt, mt = techs_for(X, gas, coal), techs_for(M, gas, coal)
        gx = {k: v[h] for k, v in xg.items() if h in v}
        gm = {k: v[h] for k, v in mg.items() if h in v}
        if not gx or not gm:
            continue
        ux = units_from(xt, gx, xc, px, clearing_price=xp[h])
        um = units_from(mt, gm, mc, pm, clearing_price=mp[h])
        dx = sum(u.output_mw for u in ux) - fo / eta
        dm = sum(u.output_mw for u in um) + fo
        cx, cmv = curve(ux), curve(um)
        res = {}
        for k, c in (("none", 0.0), ("enacted", CERT[q] * e_dec),
                     ("all_sources", CERT[q] * e_grid)):
            s = solve(cx, cmv, dx, dm, eta, cap, c)
            if s is None:
                break
            res[k] = (s[1], s[2], c * s[1])
        if len(res) < 3:
            continue
        for k, (f, e, ch) in res.items():
            agg[k]["flow_mwh"] += f
            agg[k]["em_t"] += e
            agg[k]["charge_eur"] += ch
        obs_f.append(fo)
        mod_f.append(res["enacted"][0])
    corr = (st.correlation(obs_f, mod_f)
            if len(obs_f) > 2 and st.pstdev(mod_f) > 0 and st.pstdev(obs_f) > 0 else None)
    out = dict(border=border, hours=len(obs_f),
               calibration=dict(observed_flow_gwh=sum(obs_f) / 1e3,
                                modelled_flow_enacted_gwh=sum(mod_f) / 1e3, hourly_corr=corr),
               scenarios=agg)
    f0, f1, f2 = (agg[k]["flow_mwh"] for k in ("none", "enacted", "all_sources"))
    e0, e1, e2 = (agg[k]["em_t"] for k in ("none", "enacted", "all_sources"))
    out["enacted_vs_none"] = dict(d_imports_gwh=(f1 - f0) / 1e3, d_emissions_t=e1 - e0,
                                  gross_charge_eur=agg["enacted"]["charge_eur"])
    out["allsources_vs_enacted"] = dict(d_imports_gwh=(f2 - f1) / 1e3, d_emissions_t=e2 - e1,
                                        d_charge_eur=agg["all_sources"]["charge_eur"]
                                        - agg["enacted"]["charge_eur"])
    return out


def main():
    rows = []
    raw = pathlib.Path("data/raw/gb_fr_2026")
    carbon = lambda h: (C26[str(month(h).month)]["gb_dispatch_eur"],   # noqa: E731
                        C26[str(month(h).month)]["eua_eur"])
    for imp in ("nl", "be", "fr"):
        links, ta, pa, ea = RG.IMPORTERS[imp]
        rows.append(run(f"GB->{imp.upper()}", lambda imp=imp: RG.load_gb_fr(raw, imp),
                        G.GB_TECHS, getattr(G, ta), carbon, getattr(G, ea), 0.430, 0.193, -1,
                        lambda h, imp=imp: CAP[imp]))
    rraw = pathlib.Path("data/raw/rs_hu_2026")

    def rl():
        rs, hu, rp, hp, fl, rc, hc = RY.load_rs_hu(rraw)
        return rs, rp, hu, hp, fl, rc, hc
    # RS->HU 2026 NTC not sourced: the 2025 mean monthly NTC (544 MW) is a DECLARED proxy.
    rows.append(run("RS->HU", rl, R.RS_TECHS, R.HU_TECHS,
                    lambda h: (4.0, C26[str(month(h).month)]["eua_eur"]),
                    R.ETA_LINK_RANGE, 1.041, 0.729, +1, lambda h: 544.0))
    for r in rows:
        c, a, b = r["calibration"], r["enacted_vs_none"], r["allsources_vs_enacted"]
        cr = "n/a" if c["hourly_corr"] is None else f"{c['hourly_corr']:.2f}"
        print(f"{r['border']:7s} {r['hours']:5d} h | calibration: observed {c['observed_flow_gwh']:8.1f} GWh, "
              f"modelled {c['modelled_flow_enacted_gwh']:8.1f} GWh, hourly corr {cr}")
        print(f"          enacted vs none:        dImports {a['d_imports_gwh']:+8.1f} GWh, "
              f"dCO2 {a['d_emissions_t']:+11,.0f} t, gross charge EUR {a['gross_charge_eur']:>13,.0f}")
        print(f"          all-sources vs enacted: dImports {b['d_imports_gwh']:+8.1f} GWh, "
              f"dCO2 {b['d_emissions_t']:+11,.0f} t, dCharge EUR {b['d_charge_eur']:>+13,.0f}")
    pathlib.Path("data/processed/finite_2026.json").write_text(json.dumps(rows, indent=1),
                                                               encoding="utf-8")


if __name__ == "__main__":
    main()
