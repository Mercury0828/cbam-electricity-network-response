"""Learned exporter-only rules against the network benchmark (A58; network form of Proposition 2).

Benchmark (eq. tauref-net): u = median over trained network-response models of max(0, -sum_i (s - p_i) r_i).
Group form (eq. tauref-set): zones are grouped by carbon cost, C = zones at the EU allowance price (EU, CH, NO), and
the non-EU zones RS (4 EUR/t), Z0 = BA + ME + MK (0) and GB (UKA + CPS), so that
    tau = (s - p_x) E_x - sum_G (s - p_G) M_G ,   M_G = sum_{i in G, i != x} r_i ,  E_x = -r_x.
A learned rule charges a = max(0, (s - p_x) Ehat_x - sum_G (s - p_G) Mhat_G), with the responses (seed means) learned
by gradient boosting on 2025 export hours and applied to January-June 2026:
  exporter_only     Ehat_x and Mhat_G from exporter-side information (exporter node state, border flow, calendar)
  importer_added    Ehat_x as exporter_only; Mhat_G with the importer node's state added
  full_network      Ehat_x and Mhat_G from the states of all zones
  known_Ex_*        E_x given (the setting of Proposition 2), Mhat_G from exporter-side or from all information
At s = p_C the EU-set term cancels (eq. tauref-set), so importer-side information can matter only through E_x and
the non-EU zones. Also reported: gross default, contemporaneous credit, all-sources proxy with credit, zero charge.
Valuations s in {certificate price, EU allowance price (the importer's carbon cost), 150, 200}.
Grid: share of volume on which network information lowers the hourly regret by more than 1 EUR/MWh
(|a_exporter_only - u| - |a_full_network - u| > 1), over valuations s and exporter carbon costs p_x (GB borders).
A66: path-consistent form. A rule charges every import of the exporter into the EU, so along
the response path the charges paid change by a * chi, with chi the charged-import retention of net_taxbase.py (median
over trained models). A rule that wants the charge change to equal its predicted benchmark sets a = tau_hat / chi_hat,
with chi_hat learned from the same information set as its responses; its path regret is |a chi - u|, and fixed rules
score |a chi - u|. Reported next to the single-link form (keys "path", "grid_path").
Usage: WEDGE_GNN_SPEC=v4 NETRESP_TAG=r2 python src/wedge/gnn/net_frontier.py
Output: <netresp OUTD>/net_frontier.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import SPLITS, Features, hour_of      # noqa: E402
from wedge.gnn.net_rules import E_ALL, E_DEC, cert             # noqa: E402
from wedge.gnn.netresp import OUTD                             # noqa: E402

GROUPS = {"C": None, "RS": ("RS",), "Z0": ("BA", "ME", "MK"), "GB": ("GB",)}
S_GRID = np.arange(40, 251, 10)
PX_GRID = np.arange(0, 161, 10)


def fit_predict(F, y, w, tr, seed=0):
    import xgboost as xgb
    mdl = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
                           tree_method="hist", device="cpu", n_jobs=8, random_state=seed)
    mdl.fit(F[tr], y[tr], sample_weight=w[tr])
    return mdl.predict(F)


def main():
    fe = Features()
    idx = {n: i for i, n in enumerate(fe.nodes)}
    res = {}
    for b in E_DEC:
        x, m = b.split("->")
        xi, mi = idx[x], idx[m]
        z = np.load(OUTD / f"charged_{b.replace('->', '_')}_graph.npz")
        hours, r, flow = z["hours"], z["r"].astype(np.float64), z["flow"].astype(np.float64)
        tr = (hours >= hour_of(SPLITS["y2025"][0])) & (hours < hour_of(SPLITS["y2025"][1]))
        te = (hours >= hour_of(SPLITS["charged"][0])) & (hours < hour_of(SPLITS["charged"][1]))
        p = fe.carbon[:, hours].T.astype(np.float64)                                   # (T, N)
        groups = {}
        for g, names in GROUPS.items():
            if names is None:
                mem = [i for i, n in enumerate(fe.nodes) if n not in ("GB", "RS", "BA", "ME", "MK") and i != xi]
            else:
                mem = [idx[n] for n in names if n in idx and idx[n] != xi]
            if mem:
                groups[g] = mem
        rbar = r.mean(0)                                                                # (T, N) seed mean
        Ex = -rbar[:, xi]
        M = {g: rbar[:, mem].sum(1) for g, mem in groups.items()}
        pG = {g: p[:, mem[0]] for g, mem in groups.items()}                           # one carbon cost per group
        for g, mem in groups.items():
            assert np.allclose(p[:, mem], pG[g][:, None]), g
        px = p[:, xi]
        feat_x = np.concatenate([fe.X[hours, xi], fe.MO[hours, xi], flow[:, None] / 1000.0], 1)
        feat_m = np.concatenate([feat_x, fe.X[hours, mi], fe.MO[hours, mi]], 1)
        feat_all = np.concatenate([feat_x] + [fe.X[hours, j] for j in range(fe.N) if j != xi]
                                  + [fe.MO[hours, j] for j in range(fe.N) if j != xi], 1)
        ex_hat = {"x": fit_predict(feat_x, Ex, flow, tr), "all": fit_predict(feat_all, Ex, flow, tr)}
        M_hat = {k: {g: fit_predict(F, M[g], flow, tr) for g in groups}
                 for k, F in (("x", feat_x), ("m", feat_m), ("all", feat_all))}
        tb = np.load(OUTD / f"taxbase_{b.replace('->', '_')}_graph.npz")
        assert np.array_equal(tb["hours"], hours), b
        chi = np.median(tb["chi"].astype(np.float64), 0)                              # (T,) median over models
        chi_hat = {k: np.clip(fit_predict(F, chi, flow, tr), 0.2, None)
                   for k, F in (("x", feat_x), ("m", feat_m), ("all", feat_all))}
        w = flow[te] / flow[te].sum()
        pc = cert(hours)
        pm = p[:, mi]
        vals = {"cert": np.where(np.isfinite(pc), pc, pm), "p_m": pm, "150": np.full_like(pm, 150.0),
                "200": np.full_like(pm, 200.0)}

        def bench(s, px_=None):
            pp = p.copy()
            if px_ is not None:
                pp[:, xi] = px_
            tau = -((s[:, None] - pp)[None] * r).sum(-1)
            return np.median(np.maximum(tau, 0), 0)

        def rule(s, ex, mh, px_=None):
            pxx = px if px_ is None else px_
            t = (s - pxx) * ex
            for g in groups:
                t = t - (s - pG[g]) * mh[g]
            return np.maximum(t, 0)

        o = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "groups": {g: [fe.nodes[i] for i in mem]
                                                                         for g, mem in groups.items()},
             "mae": {"E_x|exporter": float(w @ np.abs(ex_hat["x"] - Ex)[te]),
                     "E_x|all": float(w @ np.abs(ex_hat["all"] - Ex)[te])}}
        for g in groups:
            for k in ("x", "m", "all"):
                o["mae"][f"M_{g}|{k}"] = float(w @ np.abs(M_hat[k][g] - M[g])[te])
        for sname, s in vals.items():
            u = bench(s)
            sr = vals["cert"]
            rules = {"exporter_only": rule(s, ex_hat["x"], M_hat["x"]),
                     "importer_added": rule(s, ex_hat["x"], M_hat["m"]),
                     "full_network": rule(s, ex_hat["all"], M_hat["all"]),
                     "known_Ex_exporter_only": rule(s, Ex, M_hat["x"]),
                     "known_Ex_full_network": rule(s, Ex, M_hat["all"]),
                     "gross_default": sr * E_DEC[b],
                     "default_credit": np.maximum((sr - px) * E_DEC[b], 0),
                     "all_sources_credit": np.maximum((sr - px) * E_ALL[b], 0),
                     "zero": np.zeros_like(s)}
            out = {"mean_u": float(w @ u[te])}
            for k, a in rules.items():
                out[k] = float(w @ np.abs(a - u)[te])
            out["information_value"] = out["exporter_only"] - out["full_network"]
            out["information_value_known_Ex"] = out["known_Ex_exporter_only"] - out["known_Ex_full_network"]
            out["information_value_importer"] = out["exporter_only"] - out["importer_added"]
            chi_of = {"exporter_only": "x", "importer_added": "m", "full_network": "all",
                      "known_Ex_exporter_only": "x", "known_Ex_full_network": "all"}
            path = {}
            for k, a in rules.items():
                a_path = a / chi_hat[chi_of[k]] if k in chi_of else a
                path[k] = float(w @ np.abs(a_path * chi - u)[te])
            path["information_value"] = path["exporter_only"] - path["full_network"]
            path["information_value_known_Ex"] = path["known_Ex_exporter_only"] - path["known_Ex_full_network"]
            path["information_value_importer"] = path["exporter_only"] - path["importer_added"]
            path["chi_mae_exporter_only"] = float(w @ np.abs(chi_hat["x"] - chi)[te])
            path["chi_mae_full_network"] = float(w @ np.abs(chi_hat["all"] - chi)[te])
            out["path"] = path
            o[sname] = out
            print(b, sname, {k: round(v, 3) for k, v in out.items() if not isinstance(v, dict)},
                  "path", {k: round(v, 3) for k, v in out["path"].items()}, flush=True)
        if x == "GB":
            grid = np.zeros((len(PX_GRID), len(S_GRID)))
            gridv = np.zeros_like(grid)
            for i, pxv in enumerate(PX_GRID):
                pxa = np.full(len(hours), float(pxv))
                for j, sv in enumerate(S_GRID):
                    s = np.full(len(hours), float(sv))
                    u = bench(s, pxa)
                    d = np.abs(rule(s, ex_hat["x"], M_hat["x"], pxa) - u) - np.abs(rule(s, ex_hat["all"], M_hat["all"], pxa) - u)
                    grid[i, j] = float(w @ (d[te] > 1.0))
                    gridv[i, j] = float(w @ d[te])
            o["grid"] = {"s": S_GRID.tolist(), "p_x": PX_GRID.tolist(), "share_value_above_1": grid.tolist(),
                         "mean_value": gridv.tolist()}
            gp, gpv = np.zeros_like(grid), np.zeros_like(grid)
            for i, pxv in enumerate(PX_GRID):
                pxa = np.full(len(hours), float(pxv))
                for j, sv in enumerate(S_GRID):
                    s = np.full(len(hours), float(sv))
                    u = bench(s, pxa)
                    ax = rule(s, ex_hat["x"], M_hat["x"], pxa) / chi_hat["x"]
                    aa = rule(s, ex_hat["all"], M_hat["all"], pxa) / chi_hat["all"]
                    d = np.abs(ax * chi - u) - np.abs(aa * chi - u)
                    gp[i, j] = float(w @ (d[te] > 1.0))
                    gpv[i, j] = float(w @ d[te])
            o["grid_path"] = {"share_value_above_1": gp.tolist(), "mean_value": gpv.tolist()}
        res[b] = o
    (OUTD / "net_frontier.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
