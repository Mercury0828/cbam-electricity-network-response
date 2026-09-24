"""D2 re-specified (D-018, pre-registered 2026-09-21 07:14 UTC before this script was written):
a money-valued information frontier under the tau* benchmark (A33).

    tau* = (s - p_x) E_x - (s - p_m) E_m        per delivered MWh

Rule ladder (nested information sets):
  L0 s * e_dec                      constant enacted factor, gross
  L1 (s - p_x) * e_dec              + Art. 9 credit
  L2 (s - p_x) * a_x(h)             + OBSERVABLE exporter hourly average intensity (no model)
  L3 (s - p_x) * E_x(h)             exporter-side ORACLE (dispatch model; not deployable)
  L4 tau*                           two-sided oracle, gap 0

Distances |rule - tau*| are computed scenario by scenario over the 50 independent-corner scenarios
(A30) and summarised per hour as [min, max]; volume-weighted means over resolved hours.
V_hourly = gap(L1) - gap(L2); V_importer = gap(L3). Predictions P-D2a..c in D-018.
Legal charge rules always use the certificate price; s varies only in the benchmark.
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
S_GRID = (None, 150.0, 200.0)          # None = certificate price
RULES = ("L0", "L1", "L2", "L3")


def ef_mid(techs):
    return {n: sum(t.ef_el_range) / 2 for n, t in techs.items()}


def collect(loader, X, M, carbon_x, eta_rng, sign_flow, e_dec):
    xg, xp, mg, mp, flow, xc, mc = loader()
    fuel = daily_ttf_fuel(1.16)
    efx = ef_mid(X)
    rows = []
    for h in sorted(set(xp) & set(mp) & set(flow)):
        d = month(h)
        if d.year != 2026 or d.month > 6:
            continue
        f = sign_flow * flow[h]
        if f <= 0:
            continue
        cert = CERT[(d.month - 1) // 3 + 1]
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
        tot = sum(max(v, 0.0) for v in gx.values()) or 1.0
        a_x = sum(max(gx[k], 0.0) * efx.get(k, 0.0) for k in gx) / tot
        rows.append(dict(h=h, mwh=f, sc=sc, cert=cert, px=px, pm=pm, a_x=a_x, e_dec=e_dec))
    return rows


def gaps(rows, s_override):
    lab = [r for r in rows if r["sc"] is not None]
    tot = sum(r["mwh"] for r in rows)
    w = sum(r["mwh"] for r in lab)
    acc = {k: [0.0, 0.0] for k in RULES}
    for r in lab:
        s = r["cert"] if s_override is None else s_override
        per = {k: [] for k in RULES}
        for ex, im in r["sc"]:
            tau = (s - r["px"]) * ex - (s - r["pm"]) * im
            rule = {"L0": r["cert"] * r["e_dec"],
                    "L1": (r["cert"] - r["px"]) * r["e_dec"],
                    "L2": (r["cert"] - r["px"]) * r["a_x"],
                    "L3": (s - r["px"]) * ex}
            for k in RULES:
                per[k].append(abs(rule[k] - tau))
        for k in RULES:
            acc[k][0] += r["mwh"] * min(per[k])
            acc[k][1] += r["mwh"] * max(per[k])
    g = {k: (acc[k][0] / w, acc[k][1] / w) for k in RULES}
    mid = {k: sum(v) / 2 for k, v in g.items()}
    return dict(resolved_pct=100 * w / tot, gap_eur_mwh=g,
                V_hourly=mid["L1"] - mid["L2"], V_importer=mid["L3"],
                credit_value=mid["L0"] - mid["L1"])


def main():
    borders = []
    raw = pathlib.Path("data/raw/gb_fr_2026")
    for imp in ("nl", "be"):
        links, ta, pa, ea = RG.IMPORTERS[imp]
        borders.append((f"GB->{imp.upper()}", collect(
            lambda imp=imp: RG.load_gb_fr(raw, imp), G.GB_TECHS, getattr(G, ta),
            lambda h: C26[str(month(h).month)]["uka_eur"], getattr(G, ea), -1, 0.430)))
    rraw = pathlib.Path("data/raw/rs_hu_2026")

    def rl():
        rs, hu, rp, hp, fl, rc, hc = RY.load_rs_hu(rraw)
        return rs, rp, hu, hp, fl, rc, hc
    borders.append(("RS->HU", collect(rl, R.RS_TECHS, R.HU_TECHS, lambda h: 4.0,
                                      R.ETA_LINK_RANGE, +1, 1.041)))
    out = {}
    for s in S_GRID:
        tag = "s=cert" if s is None else f"s={int(s)}"
        out[tag] = {}
        for name, rows in borders:
            res = gaps(rows, s)
            out[tag][name] = res
            g = res["gap_eur_mwh"]
            print(f"{tag:7s} {name:7s} resolved {res['resolved_pct']:5.1f}% | mean |rule - tau*| EUR/MWh: "
                  + " ".join(f"{k} [{g[k][0]:5.2f},{g[k][1]:6.2f}]" for k in RULES)
                  + f" | credit value {res['credit_value']:6.2f} | V_hourly {res['V_hourly']:+6.2f}"
                  f" | V_importer {res['V_importer']:5.2f}")
    c = out["s=cert"]
    verdict = {
        "P-D2a (V_importer < 1 at s=cert, all borders)":
            all(c[b]["V_importer"] < 1 for b in c),
        "P-D2b (V_importer >= 5 at s=200 on >= 1 GB border)":
            any(out["s=200"][b]["V_importer"] >= 5 for b in out["s=200"] if b.startswith("GB")),
        "P-D2c (credit value > V_hourly on GB borders at s=cert)":
            all(c[b]["credit_value"] > c[b]["V_hourly"] for b in c if b.startswith("GB")),
    }
    for k, v in verdict.items():
        print(f"{k}: {'HOLDS' if v else 'FALSIFIED'}")
    out["preregistered_verdicts"] = verdict
    pathlib.Path("data/processed/d2_money.json").write_text(json.dumps(out, indent=1),
                                                           encoding="utf-8")


if __name__ == "__main__":
    main()
