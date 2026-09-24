"""Accounting / uncertainty audit of the benchmark (A40).

Three quantities kept separate for every hour and admissible scenario:
    tau_ref      signed economic benchmark  s(E_x - E_m) - (c_x - c_m)
    u            its non-negative projection max(0, tau_ref)   (floor applied HOURLY, per scenario,
                 BEFORE any aggregation)
    a            the evaluated charge (a >= 0)
For a >= 0 the regret relative to the constrained oracle is r(a) = |a - tau| - |u - tau| = |a - u|.

Institutional states (GB; RS unchanged throughout):
  S26   2026: GB dispatch carbon = UKA + CPS; credits "UKA+CPS" (Commission's COM(2025) 783 logic)
        and "UKA only".
  PFX   post-CPS, FIXED 2026 dispatch state (accounting experiment): CPS removed from BOTH the credit
        and the exporter carbon cost in tau_ref; E_x, E_m from the 2026 dispatch.
  PRD   post-CPS, RE-DISPATCHED: GB units re-priced at UKA only, scenarios recomputed; CPS removed
        from both sides.
Label everywhere: "default-value liability scenario on covered imports" — public flows do not show
which declarations used default values; nothing here is a charge actually paid.
Output: per state x border x rule: mean signed gap a - tau [min, max], mean regret |a - u| [min, max],
mean tau_ref [min, max], mean u [min, max], share of hours with a > max u (robustly above the feasible
oracle). s = P_C (EU-carbon-price-anchored consistency benchmark).
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as G, inputs_rs_hu_2025_12 as R      # noqa: E402
from wedge import run_year as RY, run_year_gbfr as RG                        # noqa: E402
from wedge.benchmark import C26, collect                                    # noqa: E402
from wedge.run_tiers import month                                            # noqa: E402

E = {"GB": (0.430, 0.193), "RS": (1.041, 0.729)}


def rules(r, country, credit_price):
    P = r["P_C"]
    e_dec, e_all = E[country]
    out = {"G  gross default": P * e_dec,
           "A  all-sources proxy": P * e_all,
           "C  default + credit": max(0.0, (P - credit_price) * e_dec),
           "AC all-sources proxy + credit": max(0.0, (P - credit_price) * e_all)}
    return out


SCALE = (1.0, 1.0)   # A41 sensitivity: (k_x, k_m) multiply E_x, E_m (regression-calibrated check)


def evaluate(rows, country, cx_price, credit_price):
    """cx_price(r) -> exporter carbon price in tau_ref; credit_price(r) -> credited price."""
    lab = [r for r in rows if r["sc"] is not None]
    tot = sum(r["mwh"] for r in rows)
    w = sum(r["mwh"] for r in lab)
    acc = {}
    tau_m = [0.0, 0.0]
    u_m = [0.0, 0.0]
    pos = 0.0                                  # R1-02: volume with a positive admissible tau_ref
    pos_by_month = {}
    hourly_max_regret = {}
    for r in lab:
        s = r["P_C"]
        kx, km = SCALE
        taus = [s * (kx * ex - km * im) - (cx_price(r) * kx * ex - r["p_m"] * km * im)
                for _, _, _, ex, im in r["sc"]]
        us = [max(0.0, t) for t in taus]
        mo = str(month(r["h"]).month)
        pm_ = pos_by_month.setdefault(mo, [0.0, 0.0])
        pm_[1] += r["mwh"]
        if max(taus) > 0:
            pos += r["mwh"]
            pm_[0] += r["mwh"]
        tau_m[0] += r["mwh"] * min(taus)
        tau_m[1] += r["mwh"] * max(taus)
        u_m[0] += r["mwh"] * min(us)
        u_m[1] += r["mwh"] * max(us)
        for k, a in rules(r, country, credit_price(r)).items():
            d = acc.setdefault(k, dict(sg=[0.0, 0.0], rg=[0.0, 0.0], above=0.0, mean_a=0.0))
            sg = [a - t for t in taus]
            rg = [abs(a - u) for u in us]
            d["sg"][0] += r["mwh"] * min(sg)
            d["sg"][1] += r["mwh"] * max(sg)
            d["rg"][0] += r["mwh"] * min(rg)
            d["rg"][1] += r["mwh"] * max(rg)
            d["above"] += r["mwh"] * (a > max(us))
            d["mean_a"] += r["mwh"] * a
            hourly_max_regret[k] = max(hourly_max_regret.get(k, 0.0), max(rg))   # R1-04
    return dict(
        charged_mwh=tot, resolved_mwh=w, resolved_pct=100 * w / tot,
        tau_ref=[tau_m[0] / w, tau_m[1] / w], u=[u_m[0] / w, u_m[1] / w],
        positive_target_share_pct=100 * pos / w,
        positive_target_share_by_month_pct={k: 100 * a / b for k, (a, b) in sorted(pos_by_month.items())},
        rules={k: dict(mean_charge=v["mean_a"] / w,
                       signed_gap=[v["sg"][0] / w, v["sg"][1] / w],
                       regret=[v["rg"][0] / w, v["rg"][1] / w],
                       robustly_above_feasible_oracle_pct=100 * v["above"] / w,
                       hourly_max_regret=hourly_max_regret[k])
               for k, v in acc.items()})


def main():
    global SCALE
    for a in sys.argv[1:]:
        if a.startswith("--scale="):
            SCALE = tuple(float(x) for x in a[8:].split(","))
    raw = pathlib.Path("data/raw/gb_fr_2026")
    m = lambda r: C26[str(month(r["h"]).month)]                               # noqa: E731

    def gb_rows(pdisp_key):
        def prices(h):
            c = C26[str(month(h).month)]
            return dict(p_x_dispatch=c[pdisp_key], p_x_uka=c["uka_eur"],
                        p_x_credit_full=c["gb_dispatch_eur"])
        out = {}
        for imp in ("nl", "be"):
            links, ta, pa, ea = RG.IMPORTERS[imp]
            out[f"GB->{imp.upper()}"] = collect(lambda imp=imp: RG.load_gb_fr(raw, imp), G.GB_TECHS,
                                                getattr(G, ta), prices, getattr(G, ea), -1)
        return out
    rows26 = gb_rows("gb_dispatch_eur")          # UKA + CPS in dispatch
    rowsRD = gb_rows("uka_eur")                  # re-dispatched without CPS
    rraw = pathlib.Path("data/raw/rs_hu_2026")

    def rl():
        rs, hu, rp, hp, fl, rc, hc = RY.load_rs_hu(rraw)
        return rs, rp, hu, hp, fl, rc, hc
    rs_rows = collect(rl, R.RS_TECHS, R.HU_TECHS,
                      lambda h: dict(p_x_dispatch=4.0, p_x_uka=4.0, p_x_credit_full=4.0),
                      R.ETA_LINK_RANGE, +1)
    states = {
        "S26 credit UKA+CPS": (rows26, lambda r: m(r)["gb_dispatch_eur"], lambda r: m(r)["gb_dispatch_eur"]),
        "S26 credit UKA only": (rows26, lambda r: m(r)["gb_dispatch_eur"], lambda r: m(r)["uka_eur"]),
        "PFX post-CPS fixed state": (rows26, lambda r: m(r)["uka_eur"], lambda r: m(r)["uka_eur"]),
        "PRD post-CPS re-dispatched": (rowsRD, lambda r: m(r)["uka_eur"], lambda r: m(r)["uka_eur"]),
    }
    res = {}
    for st, (rows, cx, cr) in states.items():
        res[st] = {}
        for b, rr in rows.items():
            res[st][b] = evaluate(rr, "GB", cx, cr)
    res["RS 2026"] = {"RS->HU": evaluate(rs_rows, "RS", lambda r: 4.0, lambda r: 4.0)}
    for st, bd in res.items():
        for b, o in bd.items():
            print(f"{st:28s} {b:7s} resolved {o['resolved_pct']:5.1f}% | tau_ref [{o['tau_ref'][0]:+7.2f},"
                  f"{o['tau_ref'][1]:+7.2f}] | u [{o['u'][0]:5.2f},{o['u'][1]:5.2f}]")
            for k, v in o["rules"].items():
                print(f"     {k:30s} a={v['mean_charge']:6.2f} | signed a-tau [{v['signed_gap'][0]:+7.2f},"
                      f"{v['signed_gap'][1]:+7.2f}] | regret |a-u| [{v['regret'][0]:6.2f},{v['regret'][1]:6.2f}]"
                      f" | robustly above {v['robustly_above_feasible_oracle_pct']:5.1f}%")
    pathlib.Path("data/processed/audit_benchmark" + ("" if SCALE == (1.0, 1.0) else "_scaled") +
                 ".json").write_text(json.dumps(res, indent=1),
                                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
