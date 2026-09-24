"""Money-valued benchmark for the CBAM electricity charge that respects EXISTING carbon prices (A33).

Orientation review `paper-orientation-post-audit` R1, finding B1: comparing e_dec with the physical
consequence kappa is not a tax-calibration statement, because both sides already price carbon. In a
competitive-dispatch benchmark with social carbon value s and carbon prices p_x (exporter) and p_m
(importer) already in marginal costs, the ADDITIONAL corrective charge per delivered MWh is

    tau* = s (E_x - E_m) - (p_x E_x - p_m E_m)

with E_x = e_out/eta (exporter emissions avoided per MWh of delivered import suppressed) and
E_m = e_in (importer emissions added). With s = p_m this is (s - p_x) E_x: the carbon-price
DIFFERENTIAL on the exporter's marginal emissions - which is exactly what CBAM's Art. 9 deduction is
designed to implement. The gap between the enacted GROSS charge  s * e_dec  and tau* decomposes
EXACTLY into three terms:

    gross - tau* =  s (e_dec - E_x)          factor error   (declared vs exporter marginal)
                  + p_x E_x                  exporter carbon price not deducted (no UK/RS default
                                             carbon price published -> Art. 9 unavailable, law F7)
                  + (s - p_m) E_m            importer price gap (0 when p_m = s)

Settings: s = CBAM certificate price (2026 Q1 75.36, Q2 75.28 EUR/t; law F7) -> Jan-Jun 2026 only
(Q3 unpublished). p_x = UKA monthly in EUR (GB), 4 EUR/t nominal (RS, which overstates the effective
Serbian price, law F7). p_m = EUA monthly. E_x, E_m from `reclear` over the 50 independent-corner
scenarios (A30); an hour enters only if every scenario is solved and signed. tau* and the terms are
reported as volume-weighted means of per-hour [min, max] over scenarios.

🔴 This is a stylised benchmark (competitive dispatch, marginal 1 MWh, fixed demand, no leakage
beyond the two-zone boundary). It is NOT a welfare-optimal tax claim and the paper must say so.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as G, inputs_rs_hu_2025_12 as R      # noqa: E402
from wedge import run_year as RY, run_year_gbfr as RG                        # noqa: E402
from wedge.run_year import daily_ttf_fuel, techs_for                         # noqa: E402
from wedge.reclear import units_from, robust_scenarios                        # noqa: E402
from wedge.run_tiers import C26, month                                       # noqa: E402

CERT = {1: 75.36, 2: 75.28}
# Sensitivity (A33): social carbon value s for the BENCHMARK only (the legal charge always uses the
# certificate price). None = s equals the certificate price.
S_OVERRIDE = None


def collect(loader, X, M, carbon_x, eta_rng, sign_flow, e_dec, e_all):
    xg, xp, mg, mp, flow, xc, mc = loader()
    fuel = daily_ttf_fuel(1.16)
    rows = []
    for h in sorted(set(xp) & set(mp) & set(flow)):
        d = month(h)
        if d.year != 2026 or d.month > 6:
            continue
        f = sign_flow * flow[h]
        if f <= 0:
            continue
        cert = CERT[(d.month - 1) // 3 + 1]
        s = cert if S_OVERRIDE is None else S_OVERRIDE
        px, pm = carbon_x(h), C26[str(d.month)]["eua_eur"]
        gas, coal = fuel(h)
        xt, mt = techs_for(X, gas, coal), techs_for(M, gas, coal)
        gx = {k: v[h] for k, v in xg.items() if h in v}
        gm = {k: v[h] for k, v in mg.items() if h in v}
        if not gx or not gm:
            continue
        sc = robust_scenarios(lambda q: units_from(xt, gx, xc, px, clearing_price=xp[h], q=q),
                              lambda q: units_from(mt, gm, mc, pm, clearing_price=mp[h], q=q),
                              flow_mw=f, eta_range=eta_rng)
        row = dict(h=h, mwh=f, resolved=sc is not None, gross=cert * e_dec,
                   art9_full=(cert - px) * e_dec,
                   # A33 D4 on the corrected benchmark: candidate charges per delivered MWh.
                   # 'allsources' uses the Annex II grid factor as a PROXY for the COM(2025) 989
                   # recomputation (review B4); Art.9-full deducts p_x fully (hypothetical: no
                   # UK/RS default carbon price is published, law F7).
                   rules={"enacted_gross": cert * e_dec, "allsources_proxy_gross": cert * e_all,
                          "enacted_art9_full": (cert - px) * e_dec,
                          "allsources_proxy_art9_full": (cert - px) * e_all})
        if sc is not None:
            tau = [s * (ex - im) - (px * ex - pm * im) for ex, im in sc]
            # decomposition is exact for s = cert; with S_OVERRIDE the terms are reported at s
            fac = [cert * e_dec - s * ex for ex, _ in sc]
            xcp = [px * ex for ex, _ in sc]
            imp = [(s - pm) * im for _, im in sc]
            row.update(tau=(min(tau), max(tau)), factor=(min(fac), max(fac)),
                       exp_cp=(min(xcp), max(xcp)), imp_gap=(min(imp), max(imp)))
        rows.append(row)
    return rows


def summarise(tag, rows):
    lab = [r for r in rows if r["resolved"]]
    tot = sum(r["mwh"] for r in rows)
    w = sum(r["mwh"] for r in lab)
    if not w:
        return dict(border=tag, resolved_pct=0.0)

    def m(key, i=None):
        return sum((r[key] if i is None else r[key][i]) * r["mwh"] for r in lab) / w

    def rule_stats(k):
        ins = sum(r["mwh"] for r in lab if r["tau"][0] <= r["rules"][k] <= r["tau"][1]) / w
        above = sum(r["mwh"] for r in lab if r["rules"][k] > r["tau"][1]) / w
        near = sum(r["mwh"] * (0.0 if r["tau"][0] <= r["rules"][k] <= r["tau"][1] else
                               min(abs(r["rules"][k] - r["tau"][0]), abs(r["rules"][k] - r["tau"][1])))
                   for r in lab) / w
        far = sum(r["mwh"] * max(abs(r["rules"][k] - r["tau"][0]), abs(r["rules"][k] - r["tau"][1]))
                  for r in lab) / w
        return dict(mean_charge=sum(r["mwh"] * r["rules"][k] for r in lab) / w,
                    inside_pct=100 * ins, above_pct=100 * above, below_pct=100 * (1 - ins - above),
                    mean_abs_gap_eur_mwh=(near, far))

    return dict(
        rules={k: rule_stats(k) for k in lab[0]["rules"]},
        border=tag, charged_mwh=tot, resolved_mwh=w, resolved_pct=100 * w / tot,
        gross_eur_mwh=m("gross"), art9_full_hypothetical_eur_mwh=m("art9_full"),
        tau_star_eur_mwh=(m("tau", 0), m("tau", 1)),
        gap_terms_eur_mwh=dict(factor_error=(m("factor", 0), m("factor", 1)),
                               exporter_carbon_not_deducted=(m("exp_cp", 0), m("exp_cp", 1)),
                               importer_price_gap=(m("imp_gap", 0), m("imp_gap", 1))),
        share_gross_above_tau_interval_pct=100 * sum(
            r["mwh"] for r in lab if r["gross"] > r["tau"][1]) / w,
        share_tau_negative_pct=100 * sum(r["mwh"] for r in lab if r["tau"][1] < 0) / w,
        # bounds over the FULL charged volume, unresolved hours placed at either extreme
        share_gross_above_tau_full_bounds_pct=(
            100 * sum(r["mwh"] for r in lab if r["gross"] > r["tau"][1]) / tot,
            100 * (sum(r["mwh"] for r in lab if r["gross"] > r["tau"][1]) + tot - w) / tot))


def main():
    global S_OVERRIDE
    for a in sys.argv[1:]:
        if a.startswith("--s="):
            S_OVERRIDE = float(a[4:])
    res = []
    raw = pathlib.Path("data/raw/gb_fr_2026")
    for imp in ("nl", "be"):
        links, ta, pa, ea = RG.IMPORTERS[imp]
        rows = collect(lambda imp=imp: RG.load_gb_fr(raw, imp), G.GB_TECHS, getattr(G, ta),
                       lambda h: C26[str(month(h).month)]["uka_eur"], getattr(G, ea), -1, 0.430,
                       0.193)
        res.append(summarise(f"GB->{imp.upper()}", rows))
    rraw = pathlib.Path("data/raw/rs_hu_2026")

    def rl():
        rs, hu, rp, hp, fl, rc, hc = RY.load_rs_hu(rraw)
        return rs, rp, hu, hp, fl, rc, hc
    rows = collect(rl, R.RS_TECHS, R.HU_TECHS, lambda h: 4.0, R.ETA_LINK_RANGE, +1, 1.041, 0.729)
    res.append(summarise("RS->HU", rows))
    for r in res:
        if not r.get("resolved_mwh"):
            print(f"{r['border']}: no resolved hours")
            continue
        t, g = r["tau_star_eur_mwh"], r["gap_terms_eur_mwh"]
        lo, hi = r["share_gross_above_tau_full_bounds_pct"]
        print(f"{r['border']:7s} Jan-Jun 2026 | resolved {r['resolved_pct']:5.1f}% of {r['charged_mwh']:,.0f} MWh"
              f" | gross {r['gross_eur_mwh']:6.2f} EUR/MWh | tau* [{t[0]:+6.2f}, {t[1]:+6.2f}]"
              f" | hypothetical full Art.9 {r['art9_full_hypothetical_eur_mwh']:6.2f}")
        print(f"         gross > whole tau* interval: {r['share_gross_above_tau_interval_pct']:5.1f}% of resolved"
              f" (full-volume bounds {lo:5.1f}-{hi:5.1f}%) | tau* < 0: {r['share_tau_negative_pct']:4.1f}%")
        print("         gap = factor error [{:+.2f}, {:+.2f}] + exporter carbon not deducted [{:+.2f}, {:+.2f}]"
              " + importer price gap [{:+.2f}, {:+.2f}]".format(*g["factor_error"],
                                                                 *g["exporter_carbon_not_deducted"],
                                                                 *g["importer_price_gap"]))
        for k, v in r["rules"].items():
            print(f"         {k:28s} mean {v['mean_charge']:6.2f} | inside tau* {v['inside_pct']:5.1f}%"
                  f" above {v['above_pct']:5.1f}% below {v['below_pct']:5.1f}%"
                  f" | mean |gap| [{v['mean_abs_gap_eur_mwh'][0]:5.2f}, {v['mean_abs_gap_eur_mwh'][1]:5.2f}]")
    pathlib.Path("data/processed/tau_benchmark_2026h1" +
                 ("" if S_OVERRIDE is None else f"_s{int(S_OVERRIDE)}") + ".json").write_text(json.dumps(res, indent=1),
                                                                        encoding="utf-8")


if __name__ == "__main__":
    main()
