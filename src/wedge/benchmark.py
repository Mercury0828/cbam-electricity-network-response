"""EU-carbon-price-anchored consistency benchmark tau_ref and the D4 factor-by-credit comparison (A37).

Implements the reference charge of the paper:

    tau_ref(s) = s (E_x - E_m) - (c_x - c_m),     c_x = carbon cost in exporter dispatch per MWh,
                                                  c_m = p_m E_m (EU ETS)
GB: c_x = (UKA + CPS) E_x (A37 CPS correction); RS: c_x = 4 EUR/t * E_x (nominal, law F7).
s = P_C (certificate price, primary: "EU-carbon-price-anchored consistency benchmark"); diagnostics
s = p_m (harmonised: importer term vanishes by construction), 150, 200.

Interpretation (fixed wording): under the specified short-run displacement model, tau_ref is the
additional per-MWh carbon-cost wedge that reconciles the carbon valuation of marginal cross-border
substitution with carbon costs already entering dispatch. It is NOT CBAM's legally prescribed
objective and NOT a welfare-optimal tax; a negative value is a diagnostic, not a subsidy claim.

Rules (legal charges per delivered MWh; the statutory zero floor is applied explicitly, AFTER):
  G     gross default                    P_C e_dec
  C1    default + credit at UKA          max(0, P_C e_dec - UKA e_dec)          (GB only)
  C2    default + credit at UKA+CPS      max(0, P_C e_dec - (UKA+CPS) e_dec)    (GB) / 4 e_dec (RS)
  A     all-sources proxy, gross         P_C e_all   (Annex II grid factor as PROXY, review B4)
  AC    all-sources proxy + credit C2    max(0, P_C e_all - p_x^cred e_all)
Credit D_h = p_credit * e_rule is a MODELLED statutory scenario; it is NOT assumed equal to the
marginal carbon cost c_x in dispatch.

Decomposition at s = P_C, before the floor:
  rule - tau_ref = P_C (e_rule - E_x)        declaration-factor mismatch
                 + (c_x - D_h)               credit mismatch
                 + (P_C - p_m) E_m           importer valuation / timing term

Reported per rule, volume-weighted over hours where all 50 scenarios are solved and signed (A30):
signed difference [min, max], mean absolute difference [near, far], share robustly above / inside /
below the tau_ref interval, and per-hour minimax regret across the rule set. Absolute EUR/MWh only;
no ratios (they are unstable near zero).
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as G, inputs_rs_hu_2025_12 as R      # noqa: E402
from wedge import run_year as RY, run_year_gbfr as RG                        # noqa: E402
from wedge.carbon import load_c26                                           # noqa: E402
from wedge.run_year import daily_ttf_fuel, techs_for                         # noqa: E402
from wedge.reclear import units_from, robust_scenarios                        # noqa: E402
from wedge.run_tiers import month                                            # noqa: E402

C26 = load_c26()
CERT = {1: 75.36, 2: 75.28}


def collect(loader, X, M, prices, eta_rng, sign_flow):
    """prices(h) -> dict(p_x_dispatch, p_x_uka, p_x_credit_full). Returns hourly rows with keyed
    scenarios (qx, qm, eta, E_x, E_m)."""
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
        pr = prices(h)
        pm = C26[str(d.month)]["eua_eur"]
        gas, coal = fuel(h)
        xt, mt = techs_for(X, gas, coal), techs_for(M, gas, coal)
        gx = {k: v[h] for k, v in xg.items() if h in v}
        gm = {k: v[h] for k, v in mg.items() if h in v}
        if not gx or not gm:
            continue
        sc = robust_scenarios(
            lambda q: units_from(xt, gx, xc, pr["p_x_dispatch"], clearing_price=xp[h], q=q),
            lambda q: units_from(mt, gm, mc, pm, clearing_price=mp[h], q=q),
            flow_mw=f, eta_range=eta_rng, keyed=True)
        rows.append(dict(h=h, mwh=f, sc=sc, P_C=CERT[(d.month - 1) // 3 + 1], p_m=pm, **pr))
    return rows


def tau_ref(r, ex, im, s):
    return s * (ex - im) - (r["p_x_dispatch"] * ex - r["p_m"] * im)


def rules_for(r, e_dec, e_all, gb):
    P = r["P_C"]
    out = {"G": (P * e_dec, e_dec, 0.0)}
    if gb:
        out["C1"] = (max(0.0, (P - r["p_x_uka"]) * e_dec), e_dec, r["p_x_uka"] * e_dec)
    out["C2"] = (max(0.0, (P - r["p_x_credit_full"]) * e_dec), e_dec, r["p_x_credit_full"] * e_dec)
    out["A"] = (P * e_all, e_all, 0.0)
    out["AC"] = (max(0.0, (P - r["p_x_credit_full"]) * e_all), e_all, r["p_x_credit_full"] * e_all)
    return out                          # rule -> (charge, factor used, modelled credit D_h)


def summarise(rows, e_dec, e_all, gb, s_mode):
    lab = [r for r in rows if r["sc"] is not None]
    tot = sum(r["mwh"] for r in rows)
    w = sum(r["mwh"] for r in lab)
    names = list(rules_for(lab[0], e_dec, e_all, gb))
    acc = {k: dict(sd=[0.0, 0.0], ad=[0.0, 0.0], above=0.0, inside=0.0, below=0.0, regret=0.0,
                   fac=[0.0, 0.0], cred=[0.0, 0.0], imp=[0.0, 0.0]) for k in names}
    tref = [0.0, 0.0]
    for r in lab:
        rl = rules_for(r, e_dec, e_all, gb)
        s = {"cert": r["P_C"], "pm": r["p_m"]}.get(s_mode, s_mode)
        taus = [tau_ref(r, ex, im, s) for _, _, _, ex, im in r["sc"]]
        lo, hi = min(taus), max(taus)
        tref[0] += r["mwh"] * lo
        tref[1] += r["mwh"] * hi
        # minimax regret per hour: worst case over scenarios of |rule - tau| - min_rule |rule' - tau|
        best = [min(abs(rl[k][0] - t) for k in names) for t in taus]
        for k in names:
            c, e_rule, D = rl[k]
            diffs = [c - t for t in taus]
            a = acc[k]
            a["sd"][0] += r["mwh"] * min(diffs)
            a["sd"][1] += r["mwh"] * max(diffs)
            near = 0.0 if lo <= c <= hi else min(abs(c - lo), abs(c - hi))
            a["ad"][0] += r["mwh"] * near
            a["ad"][1] += r["mwh"] * max(abs(c - lo), abs(c - hi))
            a["above" if c > hi else ("below" if c < lo else "inside")] += r["mwh"]
            a["regret"] += r["mwh"] * max(abs(c - t) - b for t, b in zip(taus, best))
            if s_mode == "cert":            # decomposition, pre-floor, at s = P_C
                fac = [r["P_C"] * (e_rule - ex) for _, _, _, ex, _ in r["sc"]]
                crd = [r["p_x_dispatch"] * ex - D for _, _, _, ex, _ in r["sc"]]
                imp = [(r["P_C"] - r["p_m"]) * im for _, _, _, _, im in r["sc"]]
                for key, v in (("fac", fac), ("cred", crd), ("imp", imp)):
                    a[key][0] += r["mwh"] * min(v)
                    a[key][1] += r["mwh"] * max(v)
    out = dict(charged_mwh=tot, resolved_mwh=w, resolved_pct=100 * w / tot,
               tau_ref_mean=[tref[0] / w, tref[1] / w], rules={})
    for k in names:
        a = acc[k]
        row = dict(signed_diff=[a["sd"][0] / w, a["sd"][1] / w],
                   mean_abs_diff=[a["ad"][0] / w, a["ad"][1] / w],
                   above_pct=100 * a["above"] / w, inside_pct=100 * a["inside"] / w,
                   below_pct=100 * a["below"] / w, minimax_regret=a["regret"] / w,
                   above_pct_full_volume_bounds=[100 * a["above"] / tot,
                                                 100 * (a["above"] + tot - w) / tot])
        if s_mode == "cert":
            row["decomposition"] = dict(factor=[a["fac"][0] / w, a["fac"][1] / w],
                                        credit=[a["cred"][0] / w, a["cred"][1] / w],
                                        importer=[a["imp"][0] / w, a["imp"][1] / w])
        out["rules"][k] = row
    return out


def main():
    borders = []
    raw = pathlib.Path("data/raw/gb_fr_2026")

    def gb_prices(h):
        m = C26[str(month(h).month)]
        return dict(p_x_dispatch=m["gb_dispatch_eur"], p_x_uka=m["uka_eur"],
                    p_x_credit_full=m["gb_dispatch_eur"])
    for imp in ("nl", "be"):
        links, ta, pa, ea = RG.IMPORTERS[imp]
        borders.append((f"GB->{imp.upper()}", collect(lambda imp=imp: RG.load_gb_fr(raw, imp),
                                                     G.GB_TECHS, getattr(G, ta), gb_prices,
                                                     getattr(G, ea), -1), 0.430, 0.193, True))
    rraw = pathlib.Path("data/raw/rs_hu_2026")

    def rl():
        rs, hu, rp, hp, fl, rc, hc = RY.load_rs_hu(rraw)
        return rs, rp, hu, hp, fl, rc, hc
    rs_prices = lambda h: dict(p_x_dispatch=4.0, p_x_uka=4.0, p_x_credit_full=4.0)   # noqa: E731
    borders.append(("RS->HU", collect(rl, R.RS_TECHS, R.HU_TECHS, rs_prices, R.ETA_LINK_RANGE, +1),
                    1.041, 0.729, False))
    res = {}
    for s_mode in ("cert", "pm", 150.0, 200.0):
        tag = f"s={s_mode}" if isinstance(s_mode, str) else f"s={int(s_mode)}"
        res[tag] = {}
        for name, rows, e_dec, e_all, gb in borders:
            o = summarise(rows, e_dec, e_all, gb, s_mode)
            res[tag][name] = o
            t = o["tau_ref_mean"]
            print(f"{tag:7s} {name:7s} resolved {o['resolved_pct']:5.1f}% | tau_ref [{t[0]:+7.2f}, {t[1]:+7.2f}]")
            for k, v in o["rules"].items():
                line = (f"      {k:3s} signed [{v['signed_diff'][0]:+7.2f},{v['signed_diff'][1]:+7.2f}]"
                        f" |abs| [{v['mean_abs_diff'][0]:5.2f},{v['mean_abs_diff'][1]:6.2f}]"
                        f" above {v['above_pct']:5.1f} in {v['inside_pct']:5.1f} below {v['below_pct']:5.1f}"
                        f" | regret {v['minimax_regret']:6.2f}")
                if "decomposition" in v:
                    dc = v["decomposition"]
                    line += (f" | factor [{dc['factor'][0]:+.1f},{dc['factor'][1]:+.1f}]"
                             f" credit [{dc['credit'][0]:+.1f},{dc['credit'][1]:+.1f}]"
                             f" importer [{dc['importer'][0]:+.1f},{dc['importer'][1]:+.1f}]")
                print(line)
    pathlib.Path("data/processed/benchmark_2026h1.json").write_text(json.dumps(res, indent=1),
                                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
