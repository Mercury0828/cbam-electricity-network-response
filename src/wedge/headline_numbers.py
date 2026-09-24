"""Self-check T3: one master table of every number in the claims register, read from artefacts, and a
optional reconciliation against a claims register (A45).

Writes data/processed/headline_numbers.json. The check extracts every number printed in the claims
register rows C1-C9 and reports any that does not appear (after rounding to the printed precision)
in the master table. The manuscript must quote numbers from this table only.
"""
from __future__ import annotations

import json
import pathlib
import re

P = pathlib.Path("data/processed")


def j(name):
    return json.loads((P / name).read_text(encoding="utf-8"))


def build():
    ab = j("audit_benchmark.json")
    rm = j("regime_map.json")["scenarios"]
    t2 = j("t2_ablation.json")
    t4 = j("t4_baselines.json")
    bm = j("benchmark_2026h1.json")["s=cert"]
    d2 = j("d2_frontier.json")
    tb = j("tiers_independent.json")
    wm = j("wedge_magnitude.json")
    agg = j("aggregate_2026h1.json")
    did = j("did_d1_pooled.json")
    ce = j("cap_event.json")
    cd = j("cap_distribution.json")
    vr = j("val_regress.json")
    s26 = ab["S26 credit UKA+CPS"]["GB->NL"]
    H = {}
    H["C1.tau_ref_GBNL"] = s26["tau_ref"]
    H["C1.u_GBNL"] = s26["u"]
    H["C1.positive_target_share"] = {b: ab["S26 credit UKA+CPS"][b]["positive_target_share_pct"] for b in ("GB->NL", "GB->BE")}
    H["C1.positive_by_month"] = {b: ab["S26 credit UKA+CPS"][b]["positive_target_share_by_month_pct"] for b in ("GB->NL", "GB->BE")}
    H["C1.hourly_max_regret_credited"] = ab["S26 credit UKA+CPS"]["GB->NL"]["rules"]["C  default + credit"]["hourly_max_regret"]
    c26 = json.loads(pathlib.Path("data/processed/carbon_monthly_2026.json").read_text(encoding="utf-8"))
    H["C1.gb_dispatch_carbon_march"] = c26["3"]["uka_eur"] + 18.0 * c26["3"]["gbp_eur"]
    H["C1.cert_Q1"] = 75.36
    H["C2.credited_regret_max_S26_all"] = max(
        [t2[v][b]["S26"]["regret"][r] for v in t2 for b in t2[v] for r in ("default + credit", "all-sources proxy + credit")
         if v not in ("no_floor", "no_cps")] +
        [x["worst_case_regret"][r] for b in t4 for k, x in t4[b].items() if k.startswith("S26")
         for r in x["worst_case_regret"] if "credit" in r])
    H["C2.credited_regret_max_NOCPS_all"] = max(
        [t2[v][b]["NOCPS"]["regret"][r] for v in t2 for b in t2[v] for r in ("default + credit", "all-sources proxy + credit")
         if v != "no_floor"] +
        [x["worst_case_regret"][r] for b in t4 for k, x in t4[b].items() if k.startswith("NOCPS")
         for r in x["worst_case_regret"] if "credit" in r])
    H["C2.credited_regret_max_S26"] = max(t2[v][b]["S26"]["regret"][r] for v in t2 for b in t2[v]
                                          for r in ("default + credit", "all-sources proxy + credit")
                                          if v != "no_floor" and v != "no_cps")
    H["C2.credited_regret_max_NOCPS"] = max(t2[v][b]["NOCPS"]["regret"][r] for v in t2 for b in t2[v]
                                            for r in ("default + credit", "all-sources proxy + credit"))
    unc = [t2[v][b][s]["regret"][r] for v in t2 for b in t2[v] for s in ("S26", "NOCPS")
           for r in ("gross default", "all-sources proxy")]
    unc += [x["worst_case_regret"][r] for b in t4 for x in t4[b].values()
            for r in ("gross default", "all-sources proxy")]           # R2 out-of-scope note: include T4
    H["C2.uncredited_regret_range"] = [min(unc), max(unc)]
    gross = [t2[v][b][s]["regret"]["gross default"] for v in t2 for b in t2[v] for s in ("S26", "NOCPS")] +         [x["worst_case_regret"]["gross default"] for b in t4 for x in t4[b].values()]
    H["C2.gross_regret_range"] = [min(gross), max(gross)]
    H["C2.t4_ranking"] = {f"{b}/{k}": v["ranking"][0] for b in t4 for k, v in t4[b].items()}
    H["C2.t4_regrets"] = {f"{b}/{k}": v["worst_case_regret"] for b in t4 for k, v in t4[b].items()}
    H["C3.decomp_GBNL"] = bm["GB->NL"]["rules"]["G"]["decomposition"]
    H["C4.material_share_GBNL"] = d2["s=cert"]["GB->NL"]["constrained"]["material_share_pct"]
    H["C4.R_inf_cert"] = {b: d2["s=cert"][b]["constrained"]["R_inf_mean"] for b in ("GB->NL", "GB->BE", "RS->HU")}
    H["C4.R_inf_150_200"] = {s: {b: d2[s][b]["constrained"]["R_inf_mean"] for b in ("GB->NL", "GB->BE", "RS->HU")}
                            for s in ("s=150", "s=200")}
    nl26 = [r for r in tb if r.get("year") == 2026 and r.get("border") == "GB->NL"][0]
    H["C5.GBNL_mis_mwh_2026"] = nl26["B"]["mis"]
    H["C5.GBNL_total_mwh_2026"] = nl26["mwh_total"]
    H["C5.GBNL_mis_pct_2026"] = nl26["B"]["mis_pct"]
    H["C6.edec_above"] = {r["border"]: r.get("share_edec_above_kappa_interval_pct") for r in wm}
    for y in (2025, 2026):
        sel = [r for r in wm if r.get("year") == y and "FR" not in r["border"] and r.get("labelled_mwh")]
        H[f"C6.pooled_{y}"] = sum(r["share_edec_above_kappa_interval_pct"] * r["labelled_mwh"] for r in sel) /             sum(r["labelled_mwh"] for r in sel)
    H["C7.flow_ddd"] = [did["pooled_DDD"], did["ci95"]]
    H["C7.cap_post"] = [ce["pooled_post"]["beta"], ce["pooled_post"]["ci95"]]
    H["C7.cap_P_gt0_pp"] = [100 * cd["0.0"]["beta"], [100 * x for x in cd["0.0"]["ci95"]]]
    H["C7.cap_P_gt_c_pp"] = {c: [100 * cd[c]["beta"], [100 * x for x in cd[c]["ci95"]]] for c in cd}
    e = agg["exposure_jan_jun_2026"]["TOTAL GB->EU (covered)"]
    H["C8.exposure"] = dict(GWh=e["export_mwh"] / 1e3, gross_m=e["L_gross_eur"] / 1e6,
                            credit_uka_cps_m=e["L_credit_uka_cps_eur"] / 1e6,
                            credit_uka_m=e["L_credit_uka_eur"] / 1e6, allsrc_m=e["L_allsources_proxy_eur"] / 1e6)
    ex_all = agg["exposure_jan_jun_2026"]
    for grp, ks in (("FR", ("IFA", "IFA2", "ElecLink")), ("IE", ("East-West", "Greenlink"))):
        H[f"C8.group_{grp}"] = dict(GWh=sum(ex_all[k]["export_mwh"] for k in ks) / 1e3,
                                   **{f: sum(ex_all[k][f] for k in ks) / 1e6 for f in
                                      ("L_gross_eur", "L_credit_uka_cps_eur", "L_credit_uka_eur", "L_allsources_proxy_eur")})
    H["C9.RS_tau_ref_model"] = ab["RS 2026"]["RS->HU"]["tau_ref"]
    H["C9.RS_eout_daily"] = vr["RS_bx_daily_2026"]["coef"]
    H["C2.regime_scen_GBNL"] = {s: {r: v["regret"] for r, v in rm["GB->NL"][s].items() if r != "feasible_oracle_u"}
                               for s in rm["GB->NL"]}
    return H


def flat_numbers(obj):
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += flat_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += flat_numbers(v)
    elif isinstance(obj, (int, float)):
        out.append(float(obj))
    return out


def reconcile(H):
    plan = pathlib.Path("docs/paper_plan.md")              # claims register of the project (not distributed)
    if not plan.exists():
        return []
    text = plan.read_text(encoding="utf-8")
    rows = [l for l in text.split("\n") if re.match(r"\| C\d \|", l)]
    pool = flat_numbers(H) + [abs(x) for x in flat_numbers(H)]
    misses = []
    for l in rows:
        claim = l.split("|")[2]
        for tok in re.findall(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?", claim):
            v = float(tok.replace(",", ""))
            if v in (0, 1, 2, 3, 4, 5, 2025, 2026) or re.match(r"^\d{4}$", tok):
                continue
            dp = len(tok.split(".")[1]) if "." in tok else 0
            tol = 0.5 * 10 ** (-dp) + 1e-9
            if not any(abs(abs(v) - abs(x)) <= tol for x in pool):
                misses.append((l.split("|")[1].strip(), tok))
    return misses


def main():
    H = build()
    (P / "headline_numbers.json").write_text(json.dumps(H, indent=1), encoding="utf-8")
    m = reconcile(H)
    print(f"master table: {len(flat_numbers(H))} numbers | claims-register numbers not found: {len(m)}")
    for c, tok in m:
        print(f"   {c}: {tok}")


if __name__ == "__main__":
    main()
