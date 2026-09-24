"""Predictive accuracy of the extended MG-STGNN against non-graph baselines (A49).

Targets: hourly emissions of dispatchable classes per node (t/h, from the predicted class allocation of the
observed residual) and the day-ahead price. Periods: validation 2023 H1, test 2023 H2 - 2024, 2025, charged 2026.
Baselines:
  avg-share   training-period mean class shares of the residual, per node and hour of day (no graph, no learning)
  xgb         per-node gradient boosting of class shares and price on the same node features (no graph)
Output: data/processed/gnn/evaluate.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import SPEC, SPLITS, Features, hour_of          # noqa: E402
from wedge.gnn.marginal import load_models                        # noqa: E402

W = 24
PERIODS = ["val", "test", "y2025", "charged"]


def gnn_predict(models, fe, hours, dev):
    src, dst = torch.tensor(fe.src, device=dev), torch.tensor(fe.dst, device=dev)
    mask = torch.tensor(fe.disp_mask, device=dev)
    E, P = [], []
    for m in models:
        m.eval()
        e_all, p_all = [], []
        with torch.no_grad():
            for i in range(0, len(hours), 256):
                hs = hours[i:i + 256]
                xs, mos, tas, ccs = (torch.tensor(np.stack(a), device=dev) for a in zip(*[fe.sample(int(t), W) for t in hs]))
                r = torch.tensor(fe.resid[:, hs].T, device=dev)
                gen, pr, _ = m(xs, mos, tas, ccs, src, dst, r, mask)
                e_all.append(((gen * torch.tensor(fe.ef_disp, device=dev)).sum(-1) * 1000).cpu().numpy())
                p_all.append(pr.cpu().numpy())
        E.append(np.concatenate(e_all))
        P.append(np.concatenate(p_all))
    return np.mean(E, 0), np.mean(P, 0) * fe.ps[:, 0] + fe.pm[:, 0]


def avg_share_predict(fe, hours):
    tr = np.arange(W, hour_of(SPLITS["train"][1]))
    hod = lambda h: h % 24                                                       # noqa: E731
    sh = np.zeros((fe.N, 24, fe.gen_disp.shape[1]))
    for n in range(fe.N):
        for k in range(24):
            hs = tr[(hod(tr) == k) & fe.disp_obs_ok[n, tr]]
            tot = fe.gen_disp[n][:, hs].sum()
            sh[n, k] = fe.gen_disp[n][:, hs].sum(1) / tot if tot > 0 else 0
    s = sh[:, hod(hours)]                                                        # (N, T, D)
    return (s * fe.resid[:, hours][:, :, None] * fe.ef_disp).sum(-1).T * 1000


def xgb_predict(fe, hours):
    import xgboost as xgb
    tr = np.arange(W, hour_of(SPLITS["train"][1]), 3)
    out_e = np.zeros((len(hours), fe.N))
    out_p = np.zeros((len(hours), fe.N))
    for n in range(fe.N):
        Xtr, Xte = fe.X[tr, n], fe.X[hours, n]
        ok = fe.disp_obs_ok[n, tr]
        if ok.sum() < 500:
            continue
        tot = fe.resid[n, tr][ok]
        e_share = np.zeros((len(hours),))
        for k in range(fe.gen_disp.shape[1]):
            y = np.where(tot > 0, fe.gen_disp[n, k, tr][ok] / np.maximum(tot, 1e-6), 0)
            if y.max() <= 0:
                continue
            mdl = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, tree_method="hist",
                                   device="cuda")
            mdl.fit(Xtr[ok], y)
            e_share += np.clip(mdl.predict(Xte), 0, 1) * fe.ef_disp[k]
        out_e[:, n] = e_share * fe.resid[n, hours] * 1000
        pk = fe.price_ok[n, tr]
        mdl = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, tree_method="hist", device="cuda")
        mdl.fit(Xtr[pk], fe.price[n, tr][pk])
        out_p[:, n] = mdl.predict(Xte)
    return out_e, out_p


def metrics(pred_e, pred_p, fe, hours):
    obs_e = (fe.gen_disp[:, :, hours] * fe.ef_disp[None, :, None]).sum(1).T * 1000
    oke = fe.disp_obs_ok[:, hours].T
    okp = fe.price_ok[:, hours].T
    obs_p = np.nan_to_num(fe.price[:, hours].T)
    rm = lambda a, b, ok: float(np.sqrt(((a - b) ** 2)[ok].mean()))             # noqa: E731
    ma = lambda a, b, ok: float(np.abs(a - b)[ok].mean())                       # noqa: E731
    per_node = {fe.nodes[n]: rm(pred_e[:, n], obs_e[:, n], oke[:, n]) for n in range(fe.N) if oke[:, n].any()}
    return dict(emis_rmse=rm(pred_e, obs_e, oke), emis_mae=ma(pred_e, obs_e, oke),
                price_rmse=rm(pred_p, obs_p, okp), price_mae=ma(pred_p, obs_p, okp),
                emis_rmse_by_node=per_node)


def main():
    dev = "cuda"
    fe = Features()
    models = load_models(fe, dev)
    nograph = load_models(fe, dev, "dropTACC")
    res = {}
    for p in PERIODS:
        s, e = (hour_of(v) for v in SPLITS[p])
        hours = np.arange(max(s, W), e)
        g_e, g_p = gnn_predict(models, fe, hours, dev)
        a_e = avg_share_predict(fe, hours)
        x_e, x_p = xgb_predict(fe, hours)
        res[p] = {"MG-STGNN-X": metrics(g_e, g_p, fe, hours),
                  "avg-share": metrics(a_e, np.nan_to_num(fe.price[:, hours - 1].T), fe, hours),
                  "xgb": metrics(x_e, x_p, fe, hours)}
        if nograph:
            n_e, n_p = gnn_predict(nograph, fe, hours, dev)
            res[p]["nograph"] = metrics(n_e, n_p, fe, hours)
        for k, v in res[p].items():
            print(p, k, {kk: round(vv, 2) for kk, vv in v.items() if not isinstance(vv, dict)}, flush=True)
    outd = pathlib.Path("data/processed/gnn" if SPEC == "v1" else f"data/processed/gnn/{SPEC}")
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "evaluate.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
