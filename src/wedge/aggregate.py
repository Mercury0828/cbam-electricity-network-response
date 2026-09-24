"""Aggregate monetary exposure and conditional emission quantities (ruling #2, item 5; A44).

Observed Jan-Jun 2026 only (certificate price published). Labels fixed by the ruling:
  L = sum_h Q_h a_h                 modelled DEFAULT-VALUE LIABILITY EXPOSURE on covered imports
                                    (not revenue paid: declarations may use actual emissions; certificates
                                    are bought in 2027; Art. 9 credits may apply)
  G = sum_h Q_h (a_h - tau_ref_h)   signed benchmark difference (resolved hours only)
G is aggregated under COMMON scenarios: the same (qx, qm, eta) applied to every hour (efficiency is a
plant property shared across hours), giving the attainable whole-period range; the sum of hourly
extremes is also reported as an outer bound.
Emissions: kappa is a marginal consequence; there is no identified trade response, so emissions are
reported only for an IMPOSED uniform 100 GWh reduction of GB->NL (resp. RS->HU) imports over resolved
hours: dCO2 = -100,000 MWh x volume-weighted kappa (sign: + = emissions rise when imports fall).
Covered GB->EU links for exposure: BritNed, Nemo, IFA, IFA2, ElecLink, Viking, East-West, Greenlink
(Ireland is in the EU customs territory). Moyle (GB->Northern Ireland) excluded (CBAM does not apply
to Northern Ireland; Commission guidance May 2026). North Sea Link (Norway) not covered.
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as G, inputs_rs_hu_2025_12 as R      # noqa: E402
from wedge import run_year as RY, run_year_gbfr as RG                        # noqa: E402
from wedge.benchmark import C26, collect                                    # noqa: E402
from wedge.did_d1 import flows                                              # noqa: E402
from wedge.run_tiers import month                                            # noqa: E402

CERT = {1: 75.36, 2: 75.28}
GB_LINKS = {"BritNed": {"Netherlands(BritNed)"}, "Nemo": {"Belgium (Nemolink)"},
            "IFA": {"France(IFA)"}, "IFA2": {"IFA2 (INTIFA2)"}, "ElecLink": {"Eleclink (INTELEC)"},
            "Viking": {"Denmark (Viking link)"}, "East-West": {"Ireland(East-West)"},
            "Greenlink": {"Ireland (Greenlink)"}}


def exposure():
    out = {}
    for name, ic in GB_LINKS.items():
        fl = flows(2026, ic)
        q = {h: v for h, v in fl.items() if v > 0 and month(h).year == 2026 and month(h).month <= 6}
        mwh = sum(q.values())
        gross = sum(v * CERT[(month(h).month - 1) // 3 + 1] * 0.430 for h, v in q.items())
        cred_uc = sum(v * max(0.0, (CERT[(month(h).month - 1) // 3 + 1]
                                    - C26[str(month(h).month)]["gb_dispatch_eur"]) * 0.430)
                      for h, v in q.items())
        cred_u = sum(v * max(0.0, (CERT[(month(h).month - 1) // 3 + 1]
                                   - C26[str(month(h).month)]["uka_eur"]) * 0.430) for h, v in q.items())
        allsrc = sum(v * CERT[(month(h).month - 1) // 3 + 1] * 0.193 for h, v in q.items())
        out[name] = dict(export_mwh=mwh, L_gross_eur=gross, L_credit_uka_cps_eur=cred_uc,
                         L_credit_uka_eur=cred_u, L_allsources_proxy_eur=allsrc)
    tot = {k: sum(v[k] for v in out.values()) for k in next(iter(out.values()))}
    out["TOTAL GB->EU (covered)"] = tot
    return out


def common_scenario_gap(rows, e_dec, credit_key=None):
    """G under common scenarios for the gross default (and optionally a credit rule)."""
    lab = [r for r in rows if r["sc"] is not None]
    by = defaultdict(float)
    lo_sum = hi_sum = 0.0
    for r in lab:
        a = r["P_C"] * e_dec
        if credit_key:
            a = max(0.0, (r["P_C"] - r[credit_key]) * e_dec)
        vals = []
        for qx, qm, eta, ex, im in r["sc"]:
            t = r["P_C"] * (ex - im) - (r["p_x_dispatch"] * ex - r["p_m"] * im)
            by[(qx, qm, eta)] += r["mwh"] * (a - t)
            vals.append(a - t)
        lo_sum += r["mwh"] * min(vals)
        hi_sum += r["mwh"] * max(vals)
    v = list(by.values())
    return dict(resolved_mwh=sum(r["mwh"] for r in lab), common_scenario_range_eur=[min(v), max(v)],
                outer_bound_eur=[lo_sum, hi_sum])


def imposed_emissions(rows, gwh=100.0):
    lab = [r for r in rows if r["sc"] is not None]
    w = sum(r["mwh"] for r in lab)
    by = defaultdict(float)
    for r in lab:
        for qx, qm, eta, ex, im in r["sc"]:
            by[(qx, qm, eta)] += r["mwh"] / w * (im - ex)       # emission change per MWh import cut
    v = list(by.values())
    return dict(dCO2_t_per_imposed_100GWh_cut=[gwh * 1000 * min(v), gwh * 1000 * max(v)])


def main():
    res = {"exposure_jan_jun_2026": exposure()}
    for k, v in res["exposure_jan_jun_2026"].items():
        print(f"{k:24s} export {v['export_mwh'] / 1e3:9,.0f} GWh | L gross EUR {v['L_gross_eur'] / 1e6:7.1f} m"
              f" | credit UKA+CPS {v['L_credit_uka_cps_eur'] / 1e6:6.2f} m | credit UKA {v['L_credit_uka_eur'] / 1e6:6.1f} m"
              f" | all-sources proxy {v['L_allsources_proxy_eur'] / 1e6:6.1f} m")
    raw = pathlib.Path("data/raw/gb_fr_2026")

    def gb_prices(h):
        m = C26[str(month(h).month)]
        return dict(p_x_dispatch=m["gb_dispatch_eur"], p_x_uka=m["uka_eur"],
                    p_x_credit_full=m["gb_dispatch_eur"])
    for imp in ("nl", "be"):
        links, ta, pa, ea = RG.IMPORTERS[imp]
        rows = collect(lambda imp=imp: RG.load_gb_fr(raw, imp), G.GB_TECHS, getattr(G, ta), gb_prices,
                       getattr(G, ea), -1)
        g = common_scenario_gap(rows, 0.430)
        gc = common_scenario_gap(rows, 0.430, "p_x_credit_full")
        em = imposed_emissions(rows)
        res[f"GB->{imp.upper()}"] = dict(gap_gross=g, gap_credit_uka_cps=gc, emissions=em)
        print(f"GB->{imp.upper()} resolved {g['resolved_mwh'] / 1e3:,.0f} GWh | G gross common-scenario EUR "
              f"[{g['common_scenario_range_eur'][0] / 1e6:.1f}, {g['common_scenario_range_eur'][1] / 1e6:.1f}] m"
              f" (outer [{g['outer_bound_eur'][0] / 1e6:.1f}, {g['outer_bound_eur'][1] / 1e6:.1f}]) | credit UKA+CPS "
              f"[{gc['common_scenario_range_eur'][0] / 1e6:.2f}, {gc['common_scenario_range_eur'][1] / 1e6:.2f}] m"
              f" | imposed 100 GWh cut: dCO2 [{em['dCO2_t_per_imposed_100GWh_cut'][0]:,.0f},"
              f" {em['dCO2_t_per_imposed_100GWh_cut'][1]:,.0f}] t")
    rraw = pathlib.Path("data/raw/rs_hu_2026")

    def rl():
        rs, hu, rp, hp, fl, rc, hc = RY.load_rs_hu(rraw)
        return rs, rp, hu, hp, fl, rc, hc
    rows = collect(rl, R.RS_TECHS, R.HU_TECHS,
                   lambda h: dict(p_x_dispatch=4.0, p_x_uka=4.0, p_x_credit_full=4.0),
                   R.ETA_LINK_RANGE, +1)
    tot_rs = sum(r["mwh"] for r in rows)
    L_rs = sum(r["mwh"] * r["P_C"] * 1.041 for r in rows)
    g = common_scenario_gap(rows, 1.041)
    em = imposed_emissions(rows)
    res["RS->HU"] = dict(export_mwh=tot_rs, L_gross_eur=L_rs, gap_gross=g, emissions=em,
                         note="model levels; model-dependent per A41")
    print(f"RS->HU export {tot_rs / 1e3:,.0f} GWh | L gross EUR {L_rs / 1e6:.1f} m | G gross common-scenario "
          f"[{g['common_scenario_range_eur'][0] / 1e6:.1f}, {g['common_scenario_range_eur'][1] / 1e6:.1f}] m"
          f" | imposed 100 GWh cut: dCO2 [{em['dCO2_t_per_imposed_100GWh_cut'][0]:,.0f},"
          f" {em['dCO2_t_per_imposed_100GWh_cut'][1]:,.0f}] t (model levels)")
    pathlib.Path("data/processed/aggregate_2026h1.json").write_text(json.dumps(res, indent=1),
                                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
