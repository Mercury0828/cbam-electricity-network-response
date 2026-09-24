"""Revision checks of the draft (A48).

1. Annual credit price. Art. 9 deductions for default-value electricity use a YEARLY default carbon price, not the
   monthly carbon cost in dispatch. The draft's credited rules used the monthly GB dispatch carbon cost p_x(t) as the
   credit price. Here the credit price is a single constant q per border and period:
     q_H1     mean of monthly UKA + CPS over January-June 2026 (a yearly price that matches the period on average)
     q_2025   UKA + CPS of December 2025 (a price fixed from the preceding year, 91.58 EUR/t)
     q_uka_H1 mean of monthly UKA only over January-June 2026
   tau_ref keeps the monthly dispatch carbon cost. Before the floor the extra gap is (p_x(t) - q) E_x (review eq.).
2. Zero-charge baseline a = 0: regret = mean u.
3. Rule regret split by the sign region of the hour's admissible reference charges:
     nonpos  every scenario tau_ref <= 0      mixed  some > 0, some <= 0      pos  every scenario tau_ref > 0
4. Base-dispatch diagnostic: share of each zone's generation that the re-cleared base case (observed flow, midpoint
   scenario) moves away from the observed technology outputs, sum |g* - g_obs| / (2 sum g_obs), volume weighted.
Output: data/processed/revision_checks.json
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as G                                   # noqa: E402
from wedge import run_year_gbfr as RG                                         # noqa: E402
from wedge.benchmark import C26, collect, daily_ttf_fuel                      # noqa: E402
from wedge.reclear import _solve_fast, units_from                             # noqa: E402
from wedge.run_tiers import month                                             # noqa: E402
from wedge.run_year import techs_for                                          # noqa: E402

E_DEC, E_ALL = 0.430, 0.193
MONTHS = [str(m) for m in range(1, 7)]
Q = {
    "q_H1 (mean UKA+CPS Jan-Jun 2026)": sum(C26[m]["gb_dispatch_eur"] for m in MONTHS) / 6,
    "q_2025 (UKA+CPS Dec 2025)": G.P_CO2_GB_EUR_PER_T,
    "q_uka_H1 (mean UKA Jan-Jun 2026)": sum(C26[m]["uka_eur"] for m in MONTHS) / 6,
}


def rules(r, qset=None):
    qset = Q if qset is None else qset
    P = r["P_C"]
    px = r["p_x_credit_full"]
    out = {"zero charge": 0.0,
           "gross default": P * E_DEC,
           "all-sources proxy": P * E_ALL,
           "default + monthly credit (draft)": max(0.0, (P - px) * E_DEC),
           "all-sources + monthly credit (draft)": max(0.0, (P - px) * E_ALL)}
    for k, q in qset.items():
        out[f"default + annual credit {k}"] = max(0.0, (P - q) * E_DEC)
        out[f"all-sources + annual credit {k}"] = max(0.0, (P - q) * E_ALL)
    return out


def evaluate(rows, qset=None):
    qset = Q if qset is None else qset
    lab = [r for r in rows if r["sc"] is not None]
    acc = {}
    reg_w = {"all": 0.0, "nonpos": 0.0, "mixed": 0.0, "pos": 0.0}
    u_acc = {k: [0.0, 0.0] for k in reg_w}
    credit_gap = {}                                   # mean (p_x - q) E_x over scenarios [min, max]
    for r in lab:
        s = r["P_C"]
        px = r["p_x_dispatch"]
        taus = [s * (ex - im) - (px * ex - r["p_m"] * im) for _, _, _, ex, im in r["sc"]]
        us = [max(0.0, t) for t in taus]
        region = "nonpos" if max(taus) <= 0 else ("pos" if min(taus) > 0 else "mixed")
        w = r["mwh"]
        for g in ("all", region):
            reg_w[g] += w
            u_acc[g][0] += w * min(us)
            u_acc[g][1] += w * max(us)
        for k, q in qset.items():
            gaps = [(px - q) * ex for _, _, _, ex, _ in r["sc"]]
            d = credit_gap.setdefault(k, [0.0, 0.0])
            d[0] += w * min(gaps)
            d[1] += w * max(gaps)
        for k, a in rules(r, qset).items():
            rg = [abs(a - u) for u in us]
            for g in ("all", region):
                d = acc.setdefault(k, {}).setdefault(g, [0.0, 0.0])
                d[0] += w * min(rg)
                d[1] += w * max(rg)
    tot = reg_w["all"]
    return dict(
        volume_share={g: reg_w[g] / tot for g in reg_w},
        u={g: [u_acc[g][0] / reg_w[g], u_acc[g][1] / reg_w[g]] for g in reg_w if reg_w[g] > 0},
        annual_credit_extra_gap={k: [v[0] / tot, v[1] / tot] for k, v in credit_gap.items()},
        regret={k: {g: [v[g][0] / reg_w[g], v[g][1] / reg_w[g]] for g in v if reg_w[g] > 0}
                for k, v in acc.items()})


def scenario_split(rows, qset=None):
    """Within each fixed scenario k: volume share with tau_k > 0, and each rule's mean regret on tau_k > 0 and on
    tau_k <= 0 hours. Returns, per rule and region, the [min, max] across scenarios (scenarios with no volume in the
    region are skipped) and the [min, max] positive share."""
    from collections import defaultdict
    lab = [r for r in rows if r["sc"] is not None]
    num = defaultdict(float)   # (k, rule, region) -> sum w*regret
    den = defaultdict(float)   # (k, region) -> sum w
    for r in lab:
        s, px = r["P_C"], r["p_x_dispatch"]
        a_r = rules(r, qset)
        for qx, qm, eta, ex, im in r["sc"]:
            k = (qx, qm, round(eta, 6))
            tau = s * (ex - im) - (px * ex - r["p_m"] * im)
            u = max(0.0, tau)
            g = "pos" if tau > 0 else "nonpos"
            den[(k, g)] += r["mwh"]
            for name, a in a_r.items():
                num[(k, name, g)] += r["mwh"] * abs(a - u)
    keys = {k for k, _ in den}
    share = [den[(k, "pos")] / (den[(k, "pos")] + den[(k, "nonpos")]) for k in keys]
    out = {"positive_share": [min(share), max(share)], "regret": {}}
    names = {n for (_, n, _) in num}
    for n in names:
        for g in ("pos", "nonpos"):
            vals = [num[(k, n, g)] / den[(k, g)] for k in keys if den[(k, g)] > 0]
            if vals:
                out["regret"].setdefault(n, {})[g] = [min(vals), max(vals)]
    return out


def base_dispatch_gap(imp):
    """Share of generation the midpoint base re-clear moves away from observed outputs (both zones)."""
    raw = pathlib.Path("data/raw/gb_fr_2026")
    links, ta, pa, ea = RG.IMPORTERS[imp]
    xg, xp, mg, mp, flow, xc, mc = RG.load_gb_fr(raw, imp)
    fuel = daily_ttf_fuel(1.16)
    eta = sum(getattr(G, ea)) / 2
    num = {"GB": 0.0, imp.upper(): 0.0}
    wsum = 0.0
    for h in sorted(set(xp) & set(mp) & set(flow)):
        d = month(h)
        if d.year != 2026 or d.month > 6:
            continue
        f = -flow[h]
        if f <= 0:
            continue
        c = C26[str(d.month)]
        gas, coal = fuel(h)
        xt, mt = techs_for(G.GB_TECHS, gas, coal), techs_for(getattr(G, ta), gas, coal)
        gx = {k: v[h] for k, v in xg.items() if h in v}
        gm = {k: v[h] for k, v in mg.items() if h in v}
        if not gx or not gm:
            continue
        ux = units_from(xt, gx, xc, c["gb_dispatch_eur"], clearing_price=xp[h])
        um = units_from(mt, gm, mc, c["eua_eur"], clearing_price=mp[h])
        demand_x = sum(u.output_mw for u in ux) - f / eta
        demand_m = sum(u.output_mw for u in um) + f
        sol = _solve_fast(ux, um, demand_x, demand_m, f, eta)
        if sol is None:
            continue
        for zone, units in (("GB", ux), (imp.upper(), um)):
            obs = sum(u.output_mw for u in units)
            if obs <= 0:
                continue
            moved = sum(abs(sol[u.name] - u.output_mw) for u in units) / (2 * obs)
            num[zone] += f * moved
        wsum += f
    return {z: v / wsum for z, v in num.items()}


def main():
    raw = pathlib.Path("data/raw/gb_fr_2026")

    def prices(h):
        c = C26[str(month(h).month)]
        return dict(p_x_dispatch=c["gb_dispatch_eur"], p_x_uka=c["uka_eur"], p_x_credit_full=c["gb_dispatch_eur"])
    res = {"annual_credit_prices": Q, "borders": {}}
    for imp in ("nl", "be"):
        links, ta, pa, ea = RG.IMPORTERS[imp]
        rows = collect(lambda imp=imp: RG.load_gb_fr(raw, imp), G.GB_TECHS, getattr(G, ta), prices,
                       getattr(G, ea), -1)
        b = f"GB->{imp.upper()}"
        res["borders"][b] = evaluate(rows)
        res["borders"][b]["scenario_split"] = sp = scenario_split(rows)
        print(b, "SCENARIO SPLIT positive share", [round(x, 3) for x in sp["positive_share"]])
        for n, v in sorted(sp["regret"].items()):
            print(f"   {n:55s}", {g: [round(x, 2) for x in rr] for g, rr in v.items()})
        res["borders"][b]["base_dispatch_moved_share"] = base_dispatch_gap(imp)
        o = res["borders"][b]
        print(b, "volume shares", {k: round(v, 3) for k, v in o["volume_share"].items()})
        print("   base dispatch moved share", {k: round(v, 3) for k, v in o["base_dispatch_moved_share"].items()})
        print("   extra gap from annual credit", {k: [round(x, 2) for x in v] for k, v in o["annual_credit_extra_gap"].items()})
        for k, v in o["regret"].items():
            print(f"   {k:55s}", {g: [round(x, 2) for x in rr] for g, rr in v.items()})
    # Post-CPS state (fixed 2026 dispatch; A40: re-dispatch identical): p_x = UKA, monthly credit = UKA.
    def prices_post(h):
        c = C26[str(month(h).month)]
        return dict(p_x_dispatch=c["uka_eur"], p_x_uka=c["uka_eur"], p_x_credit_full=c["uka_eur"])
    q_post = {"q_uka_H1 (mean UKA Jan-Jun 2026)": Q["q_uka_H1 (mean UKA Jan-Jun 2026)"]}
    res["post_cps"] = {}
    for imp in ("nl", "be"):
        links, ta, pa, ea = RG.IMPORTERS[imp]
        rows = collect(lambda imp=imp: RG.load_gb_fr(raw, imp), G.GB_TECHS, getattr(G, ta), prices_post,
                       getattr(G, ea), -1)
        b = f"GB->{imp.upper()}"
        o = res["post_cps"][b] = evaluate(rows, q_post)
        print("POST-CPS", b, "volume shares", {k: round(v, 3) for k, v in o["volume_share"].items()}, "u", o["u"]["all"])
        for k, v in o["regret"].items():
            print(f"   {k:55s}", {g: [round(x, 2) for x in rr] for g, rr in v.items()})
    pathlib.Path("data/processed/revision_checks.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
