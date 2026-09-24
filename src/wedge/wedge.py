"""The declared-versus-induced wedge in MAGNITUDE, on tier-B hours.

    kappa(h) = e_out/eta - e_in   = the operating-emission change caused by 1 MWh of delivered
                                    import (= -delta_emissions of `reclear`), tCO2/MWh
    W(h)     = e_dec - kappa(h)

🔴 Interpretation under which e_dec and kappa are comparable (required by the study design before any
magnitude of W is used): the PIGOUVIAN BENCHMARK. A border charge levied at the carbon price pi
per tonne of the import's emission consequence would charge pi * kappa per MWh; the enacted charge
is pi * e_dec. W is the gap between the factor the rule applies and the factor that benchmark would
apply, in the same units and on the same boundary B (both zones modelled). It is NOT a charge anyone
levied, NOT a welfare quantity, and pi * W is left uninterpreted. kappa is marginal (1 MWh), from a
dispatch model (tier-B inputs; shared-input limitation), and is reported as an INTERVAL over the
declared efficiency corners - varied INDEPENDENTLY on the two sides (A30) - and loss bounds; only
hours where every scenario yields a signed value enter. Signs may differ across scenarios, so this
population is wider than tier B and an interval may straddle 0.

Rules compared by volume-weighted mean absolute wedge |e_rule(h) - kappa(h)|, evaluated at the
kappa interval's nearest and farthest points:
  R-default 0.430 / 1.041 (enacted), R-allsources 0.193 / 0.729 (COM(2025) 989 denominator only),
  R-hourly-avg (exporter hourly average intensity, GB only). The oracle R-kappa is 0 by construction.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as G, inputs_rs_hu_2025_12 as R      # noqa: E402
from wedge import run_year as RY, run_year_gbfr as RG                        # noqa: E402
from wedge.run_year import daily_ttf_fuel, techs_for                         # noqa: E402
from wedge.reclear import units_from, robust_delta                            # noqa: E402
from wedge.run_tiers import C26, month                                       # noqa: E402

GB_EF_MID = {n: sum(t.ef_el_range) / 2 for n, t in G.GB_TECHS.items()}


def collect(year, mm, loader, X, M, carbon, eta_rng, sign_flow, avg_ef=None):
    xg, xp, mg, mp, flow, xc, mc = loader()
    fuel = daily_ttf_fuel(1.16)
    rows = []
    for h in sorted(set(xp) & set(mp) & set(flow)):
        d = month(h)
        if d.year != year or d.month > mm:
            continue
        f = sign_flow * flow[h]
        if f <= 0:
            continue
        gas, coal = fuel(h)
        px, pm = carbon(h)
        xt, mt = techs_for(X, gas, coal), techs_for(M, gas, coal)
        gx = {k: v[h] for k, v in xg.items() if h in v}
        gm = {k: v[h] for k, v in mg.items() if h in v}
        if not gx or not gm:
            continue
        iv = robust_delta(lambda q: units_from(xt, gx, xc, px, clearing_price=xp[h], q=q),
                          lambda q: units_from(mt, gm, mc, pm, clearing_price=mp[h], q=q),
                          flow_mw=f, eta_range=eta_rng)
        row = dict(h=h, mwh=f, resolved=iv is not None)
        if iv is not None:
            row.update(k_lo=-iv[1], k_hi=-iv[0])
            if avg_ef is not None:
                tot = sum(max(v, 0.0) for v in gx.values()) or 1.0
                row["avg"] = sum(max(gx[k], 0.0) * avg_ef.get(k, 0.0) for k in gx) / tot
        rows.append(row)
    return rows


def dist(e, lo, hi):
    """(nearest, farthest) distance from a point e to the interval [lo, hi]."""
    near = 0.0 if lo <= e <= hi else min(abs(e - lo), abs(e - hi))
    return near, max(abs(e - lo), abs(e - hi))


def summarise(tag, rows, e_dec, e_all):
    lab = [r for r in rows if r["resolved"]]
    tot = sum(r["mwh"] for r in rows)
    w = sum(r["mwh"] for r in lab)
    if not w:
        return dict(border=tag, resolved_pct=0.0, labelled_mwh=0.0)

    def mean(f):
        return sum(f(r) * r["mwh"] for r in lab) / w

    def km(r):
        return (r["k_lo"] + r["k_hi"]) / 2

    srt = sorted(lab, key=km)

    def q(p):
        acc = 0.0
        for r in srt:
            acc += r["mwh"]
            if acc >= p * w:
                return km(r)

    kbar = (mean(lambda r: r["k_lo"]), mean(lambda r: r["k_hi"]))
    out = dict(border=tag, labelled_mwh=w, resolved_pct=100 * w / tot, e_dec=e_dec,
               kappa_mean_interval=kbar,
               kappa_quantiles_mid={str(p): q(p) for p in (0.1, 0.25, 0.5, 0.75, 0.9)},
               share_kappa_negative_pct=100 * sum(r["mwh"] for r in lab if r["k_hi"] < 0) / w,
               share_kappa_below_half_edec_pct=100 * sum(r["mwh"] for r in lab
                                                         if r["k_hi"] < 0.5 * e_dec) / w,
               mean_signed_wedge_W=(e_dec - kbar[1], e_dec - kbar[0]),
               # robust statements: the rule's factor lies ABOVE the entire kappa interval
               share_edec_above_kappa_interval_pct=100 * sum(
                   r["mwh"] for r in lab if e_dec > r["k_hi"]) / w,
               share_eall_above_kappa_interval_pct=100 * sum(
                   r["mwh"] for r in lab if e_all > r["k_hi"]) / w,
               share_eall_inside_kappa_interval_pct=100 * sum(
                   r["mwh"] for r in lab if r["k_lo"] <= e_all <= r["k_hi"]) / w)
    rules = {"R-default": lambda r: e_dec, "R-allsources": lambda r: e_all}
    if all("avg" in r for r in lab):
        rules["R-hourly-avg"] = lambda r: r["avg"]
    out["mean_abs_wedge"] = {
        k: (mean(lambda r, f=f: dist(f(r), r["k_lo"], r["k_hi"])[0]),
            mean(lambda r, f=f: dist(f(r), r["k_lo"], r["k_hi"])[1]))
        for k, f in rules.items()}
    return out


def main():
    res = []
    for year, mm in ((2025, 12), (2026, 8)):
        raw = pathlib.Path(f"data/raw/gb_fr_{year}")
        for imp in ("nl", "be", "fr"):
            links, ta, pa, ea = RG.IMPORTERS[imp]
            if year == 2025:
                carbon = (lambda h, pa=pa: (G.P_CO2_GB_EUR_PER_T, getattr(G, pa)))
            else:
                carbon = (lambda h: (C26[str(month(h).month)]["gb_dispatch_eur"],
                                     C26[str(month(h).month)]["eua_eur"]))
            rows = collect(year, mm, lambda imp=imp: RG.load_gb_fr(raw, imp), G.GB_TECHS,
                           getattr(G, ta), carbon, getattr(G, ea), -1, GB_EF_MID)
            res.append(dict(year=year, **summarise(f"{year} GB->{imp.upper()}", rows,
                                                   0.430, 0.193)))
        rraw = pathlib.Path(f"data/raw/rs_hu_{year}")
        if year == 2025:
            rc = (lambda h: (R.P_CO2_RS_EUR_PER_T, R.P_CO2_HU_EUR_PER_T))
        else:
            rc = (lambda h: (4.0, C26[str(month(h).month)]["eua_eur"]))

        def rl(rraw=rraw):
            rs, hu, rp, hp, fl, rcap, hcap = RY.load_rs_hu(rraw)
            return rs, rp, hu, hp, fl, rcap, hcap
        rows = collect(year, mm, rl, R.RS_TECHS, R.HU_TECHS, rc, R.ETA_LINK_RANGE, +1)
        res.append(dict(year=year, **summarise(f"{year} RS->HU", rows, 1.041, 0.729)))
    for r in res:
        if not r.get("labelled_mwh"):
            print(f"{r['border']}: no resolved hours")
            continue
        kb = r["kappa_mean_interval"]
        print(f"{r['border']:13s} resolved {r['resolved_pct']:5.1f}% ({r['labelled_mwh']:>10,.0f} MWh)"
              f" | e_dec {r['e_dec']:.3f} | mean kappa [{kb[0]:+.3f}, {kb[1]:+.3f}]"
              f" | kappa<0 {r['share_kappa_negative_pct']:4.1f}%"
              f" | kappa<e_dec/2 {r['share_kappa_below_half_edec_pct']:5.1f}%")
        print("              kappa quantiles (mid) "
              + ", ".join(f"p{int(100 * float(p))} {v:+.3f}"
                          for p, v in r["kappa_quantiles_mid"].items()))
        print(f"              e_dec above whole kappa interval {r['share_edec_above_kappa_interval_pct']:5.1f}% | "
              f"all-sources above {r['share_eall_above_kappa_interval_pct']:5.1f}%, "
              f"inside {r['share_eall_inside_kappa_interval_pct']:5.1f}%")
        print("              mean |wedge| (near, far): "
              + "; ".join(f"{k} [{a:.3f}, {b:.3f}]" for k, (a, b) in r["mean_abs_wedge"].items()))
    pathlib.Path("data/processed/wedge_magnitude.json").write_text(json.dumps(res, indent=1),
                                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
