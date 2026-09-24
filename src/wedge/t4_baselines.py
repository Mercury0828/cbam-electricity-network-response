"""Self-check T4: does the D4 ranking survive standard alternative emission-factor methods? (A45)

Same resolved GB->NL / GB->BE hours (Jan-Jun 2026), same benchmark algebra
    tau_ref = (s - p_x) E_x - (s - p_m) E_m,   s = P_C,   u = max(0, tau_ref),   regret = |a - u|,
with (E_x, E_m) supplied by three methods:
  K  two-sided dispatch kernel (ours): 50-scenario interval (A30)
  AEF  hourly AVERAGE emission factors of each zone (generation-weighted midpoint EFs) — the common
       practitioner shortcut, known to misstate marginal responses
  REG  regression marginal factors (A41, daily resolution, 2026H1): GB e_out, and each importer's
       OWN e_in (NL and BE regressions), applied as constants (E_x = e_out / eta_mid)
Rules include the hourly exporter-average factor with credit (R1-03).
Rules: gross default, default + credit at UKA+CPS, default + credit at UKA, all-sources proxy gross,
all-sources proxy + credit UKA+CPS. States: S26 (p_x = UKA + CPS) and NOCPS (p_x = UKA).
Reported: mean regret per rule and method; the rule ranking per method.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as G                            # noqa: E402
from wedge import run_year_gbfr as RG                                  # noqa: E402
from wedge.benchmark import C26, collect                               # noqa: E402
from wedge.run_tiers import month                                      # noqa: E402

REG = json.loads(pathlib.Path("data/processed/val_regress.json").read_text(encoding="utf-8"))


def aef(gen_hour, techs):
    ef = {k: sum(t.ef_el_range) / 2 for k, t in techs.items()}
    tot = sum(max(v, 0.0) for v in gen_hour.values()) or 1.0
    return sum(max(v, 0.0) * ef.get(k, 0.0) for k, v in gen_hour.items()) / tot


def main():
    raw = pathlib.Path("data/raw/gb_fr_2026")
    e_out_reg = REG["GB_bx_daily_2026"]["coef"]
    # R1-05: each importer uses ITS OWN daily regression coefficient (BE no longer borrows NL's)
    e_in_reg_by = {"nl": -REG["NL_bm_daily_2026"]["coef"], "be": -REG["BE_bm_daily_2026"]["coef"]}
    res = {}

    def prices(h):
        m = C26[str(month(h).month)]
        return dict(p_x_dispatch=m["gb_dispatch_eur"], p_x_uka=m["uka_eur"],
                    p_x_credit_full=m["gb_dispatch_eur"])
    for imp in ("nl", "be"):
        links, ta, pa, ea = RG.IMPORTERS[imp]
        loader = lambda imp=imp: RG.load_gb_fr(raw, imp)                  # noqa: E731
        xg, xp, mg, mp, *_ = loader()
        rows = [r for r in collect(loader, G.GB_TECHS, getattr(G, ta), prices, getattr(G, ea), -1)
                if r["sc"] is not None]
        eta_mid = sum(getattr(G, ea)) / 2
        w = sum(r["mwh"] for r in rows)
        out = {}
        for state in ("S26", "NOCPS"):
            for method in ("K", "AEF", "REG"):
                acc = {}
                for r in rows:
                    m = C26[str(month(r["h"]).month)]
                    P, pm = r["P_C"], r["p_m"]
                    px = m["gb_dispatch_eur"] if state == "S26" else m["uka_eur"]
                    if method == "K":
                        pairs = [(ex, im) for _, _, _, ex, im in r["sc"]]
                    elif method == "AEF":
                        gx = {k: v[r["h"]] for k, v in xg.items() if r["h"] in v}
                        gm = {k: v[r["h"]] for k, v in mg.items() if r["h"] in v}
                        pairs = [(aef(gx, G.GB_TECHS) / eta_mid, aef(gm, getattr(G, ta)))]
                    else:
                        pairs = [(e_out_reg / eta_mid, e_in_reg_by[imp])]
                    us = [max(0.0, (P - px) * ex - (P - pm) * im) for ex, im in pairs]
                    credit = m["gb_dispatch_eur"] if state == "S26" else m["uka_eur"]
                    gxh = {k: v[r["h"]] for k, v in xg.items() if r["h"] in v}
                    a_x = aef(gxh, G.GB_TECHS)          # R1-03: hourly exporter average factor
                    rules = {"gross default": P * 0.430,
                             "default + credit (exporter carbon cost)": max(0.0, (P - credit) * 0.430),
                             "hourly average factor + credit": max(0.0, (P - credit) * a_x),
                             "all-sources proxy": P * 0.193,
                             "all-sources proxy + credit": max(0.0, (P - credit) * 0.193)}
                    for k, a in rules.items():
                        acc[k] = acc.get(k, 0.0) + r["mwh"] * max(abs(a - u) for u in us)
                reg = {k: v / w for k, v in acc.items()}
                out[f"{state}/{method}"] = dict(worst_case_regret=reg,
                                                ranking=sorted(reg, key=reg.get))
                print(f"GB->{imp.upper()} {state:5s} {method:3s} | " +
                      " | ".join(f"{k}: {v:5.2f}" for k, v in reg.items()) +
                      f" | best: {sorted(reg, key=reg.get)[0]}")
        res[f"GB->{imp.upper()}"] = out
    pathlib.Path("data/processed/t4_baselines.json").write_text(json.dumps(res, indent=1),
                                                                encoding="utf-8")


if __name__ == "__main__":
    main()
