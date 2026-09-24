"""Parameterised information-regime map and forward scenarios (A42).

For each resolved export hour h (Jan-Jun 2026) and admissible scenario (qx, qm, eta, E_x, E_m):
    tau_ref(s, p_x) = (s - p_x) E_x - (s - p_m) E_m        (p_m = monthly EUA, observed)
Exporter-side oracle state = (qx, eta); importer state qm unknown.
    u = max(0, tau_ref);  R_inf(h, qx, eta) = (max_qm u - min_qm u) / 2
Grid: s in 50..250 step 10 EUR/t (social carbon valuation), p_x in 0..150 step 10 EUR/t (carbon cost in
the exporter's dispatch). Per cell, volume-weighted over resolved hours, using the WORST exporter state
per hour (stated explicitly; not a population minimax):
  zero_action_share   share of volume where every admissible tau_ref <= 0 (constrained oracle = 0)
  material_share      share of volume where R_inf > EUR 1/MWh (importer information material)
  mean_R              mean R_inf;  mean_u  mean of the hour's max u (feasible oracle charge, upper)
Regions: (i) s = p_m cancellation (importer term vanishes); (ii) common zero action; (iii) non-trivial
importer-information region, R = |s - p_m|/2 (E_m^max - E_m^min) where the floor does not bind.

Forward scenarios (fixed 2026 dispatch states; A40 shows GB re-dispatch without CPS is identical):
  S26    2026 as observed: s = quarterly P_C, p_x = UKA + CPS (GB) / 4 (RS)
  W27    2027 pricing convention: weekly certificate averaging -> s tracks the EU price in dispatch;
         proxy s = p_m (monthly EUA), p_x as S26
  NOCPS  CPS removed (announced from April 2028): p_x = UKA, s = P_C
  CONV   hypothetical UK-EU carbon-price convergence: p_x = p_m = s (price convergence only; a linkage
         agreement's LEGAL treatment of CBAM coverage is a separate question)
Rules (charges): gross default P_C e_dec; default + credit max(0, (P_C - p_credit) e_dec) with p_credit =
p_x of the scenario; all-sources proxy gross; all-sources proxy + credit. Reported: mean charge, regret
|a - u| [min, max], share robustly above u. RS appears with the MODEL levels only and is flagged
model-dependent (A41).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as G, inputs_rs_hu_2025_12 as R      # noqa: E402
from wedge import run_year as RY, run_year_gbfr as RG                        # noqa: E402
from wedge.benchmark import C26, collect                                    # noqa: E402
from wedge.run_tiers import month                                            # noqa: E402

S_GRID = np.arange(50, 251, 10, dtype=float)
PX_GRID = np.arange(0, 151, 10, dtype=float)
E = {"GB": (0.430, 0.193), "RS": (1.041, 0.729)}


def arrays(rows):
    """Per resolved hour: mwh, p_m, P_C, uka, gb_dispatch, and scenario arrays grouped by exporter
    state: EX[k], EM[k] of shape (n_states, n_importer)."""
    out = []
    for r in rows:
        if r["sc"] is None:
            continue
        groups = {}
        for qx, qm, eta, ex, im in r["sc"]:
            groups.setdefault((qx, eta), []).append((ex, im))
        ex = np.array([[a for a, _ in v] for v in groups.values()])
        em = np.array([[b for _, b in v] for v in groups.values()])
        m = C26[str(month(r["h"]).month)]
        out.append(dict(mwh=r["mwh"], p_m=r["p_m"], P_C=r["P_C"], uka=m["uka_eur"],
                        disp=r["p_x_dispatch"], ex=ex, em=em))
    return out


def cell(hours, s, px):
    tot = sum(h["mwh"] for h in hours)
    zero = mat = R_sum = u_sum = 0.0
    for h in hours:
        tau = (s - px) * h["ex"] - (s - h["p_m"]) * h["em"]
        u = np.maximum(tau, 0.0)
        Rst = (u.max(axis=1) - u.min(axis=1)) / 2.0         # per exporter state
        Rw = float(Rst.max())                               # worst exporter state
        zero += h["mwh"] * (tau.max() <= 0)
        mat += h["mwh"] * (Rw > 1.0)
        R_sum += h["mwh"] * Rw
        u_sum += h["mwh"] * float(u.max())
    return dict(zero_action_share=zero / tot, material_share=mat / tot, mean_R=R_sum / tot,
                mean_u_max=u_sum / tot)


def scenario(hours, country, name):
    e_dec, e_all = E[country]
    acc = {}
    for h in hours:
        P, pm = h["P_C"], h["p_m"]
        if country == "RS":
            s, px = P, 4.0
        elif name == "S26":
            s, px = P, h["disp"]
        elif name == "W27":
            s, px = pm, h["disp"]
        elif name == "NOCPS":
            s, px = P, h["uka"]
        else:                                               # CONV
            s, px = pm, pm
        tau = (s - px) * h["ex"] - (s - pm) * h["em"]
        u = np.maximum(tau, 0.0).ravel()
        cert = s if name in ("W27", "CONV") else P
        rules = {"gross default": cert * e_dec,
                 "default + credit": max(0.0, (cert - px) * e_dec),
                 "all-sources proxy": cert * e_all,
                 "all-sources proxy + credit": max(0.0, (cert - px) * e_all)}
        for k, a in rules.items():
            d = acc.setdefault(k, [0.0, 0.0, 0.0, 0.0, 0.0])
            reg = np.abs(a - u)
            d[0] += h["mwh"] * a
            d[1] += h["mwh"] * reg.min()
            d[2] += h["mwh"] * reg.max()
            d[3] += h["mwh"] * (a > u.max())
            d[4] += h["mwh"]
        acc.setdefault("_u", [0.0, 0.0, 0.0])
        acc["_u"][0] += h["mwh"] * u.min()
        acc["_u"][1] += h["mwh"] * u.max()
        acc["_u"][2] += h["mwh"]
    out = {k: dict(mean_charge=v[0] / v[4], regret=[v[1] / v[4], v[2] / v[4]],
                   robustly_above_pct=100 * v[3] / v[4]) for k, v in acc.items() if k != "_u"}
    out["feasible_oracle_u"] = [acc["_u"][0] / acc["_u"][2], acc["_u"][1] / acc["_u"][2]]
    return out


def main():
    raw = pathlib.Path("data/raw/gb_fr_2026")

    def gb_prices(h):
        m = C26[str(month(h).month)]
        return dict(p_x_dispatch=m["gb_dispatch_eur"], p_x_uka=m["uka_eur"],
                    p_x_credit_full=m["gb_dispatch_eur"])
    data = {}
    for imp in ("nl", "be"):
        links, ta, pa, ea = RG.IMPORTERS[imp]
        data[f"GB->{imp.upper()}"] = ("GB", arrays(collect(lambda imp=imp: RG.load_gb_fr(raw, imp),
                                                           G.GB_TECHS, getattr(G, ta), gb_prices,
                                                           getattr(G, ea), -1)))
    rraw = pathlib.Path("data/raw/rs_hu_2026")

    def rl():
        rs, hu, rp, hp, fl, rc, hc = RY.load_rs_hu(rraw)
        return rs, rp, hu, hp, fl, rc, hc
    data["RS->HU (model levels)"] = ("RS", arrays(collect(
        rl, R.RS_TECHS, R.HU_TECHS, lambda h: dict(p_x_dispatch=4.0, p_x_uka=4.0, p_x_credit_full=4.0),
        R.ETA_LINK_RANGE, +1)))
    res = {"grid": {"s": S_GRID.tolist(), "p_x": PX_GRID.tolist()}, "map": {}, "scenarios": {}}
    for b, (country, hours) in data.items():
        res["map"][b] = [[cell(hours, s, px) for px in PX_GRID] for s in S_GRID]
        res["scenarios"][b] = {n: scenario(hours, country, n)
                               for n in (("S26",) if country == "RS" else ("S26", "W27", "NOCPS", "CONV"))}
        # compact console view: material_share at a few cells
        print(f"== {b}: share of volume with material importer information (R > 1), rows s, cols p_x")
        print("      " + " ".join(f"{int(p):>5d}" for p in PX_GRID[::3]))
        for i, s in enumerate(S_GRID[::4]):
            row = res["map"][b][list(S_GRID).index(s)]
            print(f"s={int(s):3d} " + " ".join(f"{row[j]['material_share']:5.2f}" for j in range(0, len(PX_GRID), 3)))
        print(f"   zero-action share at (s=80, p_x=90): "
              f"{res['map'][b][list(S_GRID).index(80.0)][list(PX_GRID).index(90.0)]['zero_action_share']:.2f}")
        for n, sc in res["scenarios"][b].items():
            u = sc["feasible_oracle_u"]
            print(f"   {n:6s} u [{u[0]:6.2f},{u[1]:6.2f}] | " + " | ".join(
                f"{k}: a={v['mean_charge']:.1f} regret [{v['regret'][0]:.1f},{v['regret'][1]:.1f}]"
                for k, v in sc.items() if k != "feasible_oracle_u"))
    pathlib.Path("data/processed/regime_map.json").write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main()
