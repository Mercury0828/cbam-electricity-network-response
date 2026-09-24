"""Self-check T2: ablation of pipeline components on the headline D4 result (A45).

Headline: GB->NL / GB->BE regret |a - u| per rule, states S26 (2026) and NOCPS, Jan-Jun 2026.
Variants (each run in a fresh process so module-level switches take effect):
  base          as frozen (CPS on, independent corners, FUELHH repair, zero floor)
  no_cps        carbon.CPS_ON = False (GB dispatch + benchmark at UKA only: the pre-A37 accounting)
  joint         joint efficiency corners (robust_scenarios independent=False)
  exclude       data repair mode 'exclude' (drop every flagged GB hour instead of FUELHH replacement)
  no_floor      compare rules with the SIGNED tau_ref (|a - tau|) instead of the projection u
Reported: mean worst-case regret per rule and the best rule, per variant.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

VARIANTS = ["base", "no_cps", "joint", "exclude", "no_floor"]

WORKER = r'''
import sys, json, pathlib
sys.path.insert(0, "src")
variant = sys.argv[1]
import wedge.carbon as CB
if variant == "no_cps":
    CB.CPS_ON = False
import wedge.run_year_gbfr as RG
if variant == "exclude":
    _orig = RG.load_gb_fr
    RG.load_gb_fr = lambda raw, imp="fr", repair="exclude": _orig(raw, imp, repair="exclude")
import wedge.reclear as RC
if variant == "joint":
    _rs = RC.robust_scenarios
    RC.robust_scenarios = lambda *a, **k: _rs(*a, **{**k, "independent": False})
import wedge.benchmark as BM
if variant == "joint":
    BM.robust_scenarios = RC.robust_scenarios
from wedge import inputs_gb_fr_2025_12 as G
from wedge.run_tiers import month
C26 = BM.C26
raw = pathlib.Path("data/raw/gb_fr_2026")
out = {}
for imp in ("nl", "be"):
    links, ta, pa, ea = RG.IMPORTERS[imp]
    def prices(h):
        m = C26[str(month(h).month)]
        return dict(p_x_dispatch=m["gb_dispatch_eur"], p_x_uka=m["uka_eur"], p_x_credit_full=m["gb_dispatch_eur"])
    rows = [r for r in BM.collect(lambda imp=imp: RG.load_gb_fr(raw, imp), G.GB_TECHS, getattr(G, ta), prices,
                                  getattr(G, ea), -1) if r["sc"] is not None]
    w = sum(r["mwh"] for r in rows)
    res = {}
    for state in ("S26", "NOCPS"):
        acc = {}
        for r in rows:
            m = C26[str(month(r["h"]).month)]
            P, pm = r["P_C"], r["p_m"]
            px = m["gb_dispatch_eur"] if state == "S26" else m["uka_eur"]
            taus = [(P - px) * ex - (P - pm) * im for _, _, _, ex, im in r["sc"]]
            tgt = taus if variant == "no_floor" else [max(0.0, t) for t in taus]
            rules = {"gross default": P * 0.430, "default + credit": max(0.0, (P - px) * 0.430),
                     "all-sources proxy": P * 0.193, "all-sources proxy + credit": max(0.0, (P - px) * 0.193)}
            for k, a in rules.items():
                acc[k] = acc.get(k, 0.0) + r["mwh"] * max(abs(a - t) for t in tgt)
        reg = {k: v / w for k, v in acc.items()}
        res[state] = dict(regret=reg, best=min(reg, key=reg.get), resolved_mwh=w)
    out["GB->" + imp.upper()] = res
print(json.dumps(out))
'''


def main():
    res = {}
    for v in VARIANTS:
        r = subprocess.run([sys.executable, "-c", WORKER, v], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(v, "FAILED", r.stderr[-800:])
            return 1
        res[v] = json.loads(r.stdout.strip().splitlines()[-1])
        for b, st in res[v].items():
            for s, o in st.items():
                print(f"{v:9s} {b:7s} {s:5s} resolved {o['resolved_mwh'] / 1e3:7,.0f} GWh | "
                      + " | ".join(f"{k}: {x:5.2f}" for k, x in o["regret"].items()) + f" | best: {o['best']}")
    pathlib.Path("data/processed/t2_ablation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
