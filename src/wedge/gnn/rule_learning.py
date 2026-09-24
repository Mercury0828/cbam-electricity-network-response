"""Learned exporter-only rule versus the information bound (A49, part 2).

Benchmark. For each charged export hour, the graph model gives draws of (E_x, E_m) (seeds x MC dropout). With the
monthly dispatch carbon costs p_x, p_m and a valuation s,
    tau = (s - p_x) E_x - (s - p_m) E_m,   u = max(0, tau)        (benchmark = median over draws of u)
Learned rules (structure of Proposition 2). A rule sets a = max(0, (s - p_x) Ehat_x - (s - p_m) Ehat_m), where the
responses are learned by gradient boosting on 2025 export hours and applied to the charged period (2026 H1):
  Ehat_x          learned from exporter-side information for both rules (the exporter side is known, as in Prop. 2):
                  exporter node features, border flow, calendar
  Ehat_m          exporter-only rule:  learned from the same exporter-side information
                  full-graph rule:     learned with the importer node's features added
  full-graph rule, both sides (A50): Ehat_x as well learned with the importer's features, so importer
                  information may also improve the exporter-side prediction (not covered by Proposition 2)
The comparison of the first two rules holds Ehat_x fixed, so at s = p_m their equality is a mechanism check that
follows by construction; the third rule measures the value of importer information for Ehat_x.
Physical responses do not depend on the carbon-price regime, so training on 2025 and testing on 2026 does not face
the shift that a direct regression of the charge would. Also reported: gross default, contemporaneous credit, zero
charge, and the Proposition 2 bound rho* = (max u - min u)/2 over the importer draws with E_x at its median.
Valuations s in {certificate price, p_m, 150, 200}.
Output: data/processed/gnn/rule_learning.json
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import CERT_Q, SPEC, SPLITS, Features, hour_of          # noqa: E402

OUTD = "data/processed/gnn" if SPEC == "v1" else f"data/processed/gnn/{SPEC}"
E_DEC = {"GB->NL": 0.430, "GB->BE": 0.430, "RS->HU": 1.041}


def cert(hours):
    out = []
    for h in hours:
        dt = datetime.fromtimestamp(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() + 3600 * int(h), tz=timezone.utc)
        out.append(CERT_Q.get((dt.year, (dt.month - 1) // 3 + 1), np.nan))
    return np.array(out)


def fit_predict(F, y, w, tr):
    import xgboost as xgb
    mdl = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, tree_method="hist",
                           device="cuda")
    mdl.fit(F[tr], y[tr], sample_weight=w[tr])
    return mdl.predict(F)


def main():
    fe = Features()
    idx = {n: i for i, n in enumerate(fe.nodes)}
    cal = slice(10, None)                                     # calendar and regime columns of the node features
    res = {}
    for b, edec in E_DEC.items():
        f = np.load(f"{OUTD}/marginal_{b.replace('->', '_')}.npz")
        hours, draws, flow = f["hours"], f["draws"], f["flow"]            # draws: (D, T, 3)
        x, m = b.split("->")
        px, pm = fe.carbon[idx[x], hours], fe.carbon[idx[m], hours]
        pc = cert(hours)
        ex_med, em_med = np.median(draws[:, :, 0], 0), np.median(draws[:, :, 1], 0)
        feat_x = np.concatenate([fe.X[hours, idx[x]], fe.MO[hours, idx[x]], flow[:, None] / 1000.0], 1)
        feat_full = np.concatenate([feat_x, fe.X[hours, idx[m]], fe.MO[hours, idx[m]]], 1)
        tr = (hours >= hour_of(SPLITS["y2025"][0])) & (hours < hour_of(SPLITS["y2025"][1]))
        te = (hours >= hour_of(SPLITS["charged"][0])) & (hours < hour_of(SPLITS["charged"][1]))
        ex_hat = fit_predict(feat_x, ex_med, flow, tr)
        em_hat_x = fit_predict(feat_x, em_med, flow, tr)
        em_hat_f = fit_predict(feat_full, em_med, flow, tr)
        ex_hat_f = fit_predict(feat_full, ex_med, flow, tr)
        w = flow[te]
        wm = lambda a: float((a[te] * w).sum() / w.sum())                              # noqa: E731
        res[b] = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
                  "mae_E_x_hat": wm(np.abs(ex_hat - ex_med)),
                  "mae_E_m_hat_exporter_only": wm(np.abs(em_hat_x - em_med)),
                  "mae_E_m_hat_full_graph": wm(np.abs(em_hat_f - em_med)),
                  "mae_E_x_hat_full_graph": wm(np.abs(ex_hat_f - ex_med))}
        vals = {"cert": np.where(np.isfinite(pc), pc, pm), "p_m": pm, "150": np.full_like(pm, 150.0),
                "200": np.full_like(pm, 200.0)}
        for sname, s in vals.items():
            u = np.median(np.maximum((s - px)[None] * draws[:, :, 0] - (s - pm)[None] * draws[:, :, 1], 0), 0)
            u_fix = np.maximum((s - px)[None] * ex_med[None] - (s - pm)[None] * draws[:, :, 1], 0)
            rho = 0.5 * (u_fix.max(0) - u_fix.min(0))
            a_x = np.maximum((s - px) * ex_hat - (s - pm) * em_hat_x, 0)
            a_f = np.maximum((s - px) * ex_hat - (s - pm) * em_hat_f, 0)
            a_ff = np.maximum((s - px) * ex_hat_f - (s - pm) * em_hat_f, 0)
            sr = vals["cert"]
            e_all = {"GB->NL": 0.193, "GB->BE": 0.193, "RS->HU": 0.729}[b]
            q_ann = {"GB->NL": 81.25, "GB->BE": 81.25, "RS->HU": 4.0}[b]           # H1-2026 mean UKA+CPS (GB)
            out_rules = {"all_sources_proxy": wm(np.abs(sr * e_all - u)),
                         "all_sources_contemporaneous_credit": wm(np.abs(np.maximum((sr - px) * e_all, 0) - u)),
                         "default_annual_credit": wm(np.abs(np.maximum((sr - q_ann) * edec, 0) - u))}
            out = {"mean_u": wm(u), "mean_rho_star": wm(rho), **out_rules,
                   "exporter_only_learned": wm(np.abs(a_x - u)), "full_graph_learned": wm(np.abs(a_f - u)),
                   "information_value": wm(np.abs(a_x - u)) - wm(np.abs(a_f - u)),
                   "full_graph_both_learned": wm(np.abs(a_ff - u)),
                   "information_value_both": wm(np.abs(a_x - u)) - wm(np.abs(a_ff - u)),
                   "gross_default": wm(np.abs(sr * edec - u)),
                   "default_contemporaneous_credit": wm(np.abs(np.maximum((sr - px) * edec, 0) - u)),
                   "zero_charge": wm(u)}
            res[b][sname] = out
            print(b, sname, {k: round(v, 3) for k, v in out.items()}, flush=True)
        print(b, {k: round(v, 3) for k, v in res[b].items() if k.startswith("mae")}, flush=True)
    pathlib.Path(f"{OUTD}/rule_learning.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
