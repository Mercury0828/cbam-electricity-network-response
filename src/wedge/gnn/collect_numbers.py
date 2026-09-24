"""Collect every headline number of the network version (A53) from the frozen v4 artefacts into one report.

Usage: python src/wedge/gnn/collect_numbers.py [suffix]   (suffix defaults to _v4)
Output: data/processed/gnn/headline_numbers<suffix>.json plus a readable print-out. Nothing is computed here; the
script only reads artefacts, so the manuscript can quote it without re-deriving anything.
"""
from __future__ import annotations

import json
import pathlib
import sys

P = pathlib.Path("data/processed/gnn")


def jload(name):
    f = P / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def main(sfx="_v4"):
    out = {}
    did = jload(f"outage_did{sfx}.json")
    did_late = jload(f"outage_did_late{sfx}.json")
    if did:
        out["outage_did"] = {b: {k: {"did": round(v["did"], 3), "ci": [round(c, 3) for c in v["did_ci95"]]}
                                 for k, v in r.items() if isinstance(v, dict)}
                             for b, r in did.items()}
        out["outage_events"] = {b: {"events": r["events"], "pseudo": r["pseudo_events"]} for b, r in did.items()}
    if did_late:
        out["outage_did_late"] = {b: {k: {"did": round(v["did"], 3), "ci": [round(c, 3) for c in v["did_ci95"]]}
                                      for k, v in r.items() if isinstance(v, dict)}
                                  for b, r in did_late.items()}
    for tag, key in ((f"netresp_r2{sfx}/validate_outages.json", "validate_all"),
                     (f"netresp_r2{sfx}/validate_outages_late.json", "validate_late"),
                     (f"netresp_r2{sfx}/net_rules.json", "net_rules"),
                     (f"{sfx.strip('_')}/evaluate.json", "accuracy"),
                     (f"{sfx.strip('_')}/response_check.json", "response_calibration"),
                     (f"{sfx.strip('_')}/gnn_marginal_summary.json", "two_zone_marginal"),
                     (f"{sfx.strip('_')}/rule_learning.json", "learned_rules")):
        d = jload(tag)
        if d is not None:
            out[key] = d
    cmp_ = jload("compare_outage.json")
    if cmp_:
        out["compare_outage"] = cmp_
    cmpl = jload("compare_outage_late.json")
    if cmpl:
        out["compare_outage_late"] = cmpl
    (P / f"headline_numbers{sfx}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    # readable summary
    if "outage_did" in out:
        for b, r in out["outage_did"].items():
            ks = [k for k in ("R_m", "NIother_m", "R_x", "NIother_x") if k in r]
            print(b, "events", out["outage_events"][b]["events"],
                  {k: (r[k]["did"], r[k]["ci"]) for k in ks})
            print("   emissions", {k[2:]: (r[k]["did"], r[k]["ci"]) for k in r if k.startswith("E_")})
    if "accuracy" in out:
        for p, v in out["accuracy"].items():
            print("accuracy", p, {k: round(vv["emis_rmse"]) for k, vv in v.items()})
    if "net_rules" in out:
        for k, v in out["net_rules"].items():
            if "2026H1" in v:
                r = v["2026H1"]
                print("rules", k, {kk: round(vv, 3) for kk, vv in r.items() if kk.startswith(("mean_u", "regret_gross",
                                   "regret_all_sources", "regret_default_credit", "regret_default_annual",
                                   "regret_zero", "term_"))})
    print("written", P / f"headline_numbers{sfx}.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_v4")
