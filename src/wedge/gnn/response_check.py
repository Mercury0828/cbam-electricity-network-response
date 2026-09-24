"""Response-level validation of the marginal emission responses (A50).

Emission-level accuracy does not test d E / d R, the object the charge benchmark needs. This script tests the local
response to a change in a node's residual demand against observed hour-to-hour changes in held-out data (2025 and
January-June 2026), in the spirit of the empirical marginal-emission literature: a change in the border flow and a
change in residual demand from load or renewables both change the output that dispatchable plants must supply.

For each node n in {GB, NL, BE, RS, HU} and hour t with observations at t-1 and t:
  observed   dE_t = E_t - E_{t-1} (dispatchable-plant emissions, t/h),  dR_t = R_t - R_{t-1} (MW)
  model      m_t  = [E(R_t + 100 MW) - E(R_t)] / 100, all other inputs fixed; mbar_t = (m_t + m_{t-1}) / 2
Estimators of the response compared:
  graph        extended graph model (seeds averaged, eval mode)
  nograph      same architecture trained without cross-node messages
  trees        per-node boosted trees of class shares (finite difference in the residual input)
  hourly_mef   classic empirical marginal factor: OLS slope of dE on dR by node x hour-of-day x season on 2019-2022
  average      current average emission rate E_t / R_t
Tests: (i) calibration, the OLS slope of dE on mbar_t * dR (1 = calibrated), and the out-of-sample R^2 of
mbar_t * dR for dE; (ii) binned calibration, within quintiles of mbar_t the empirical slope of dE on dR.
Output: data/processed/gnn/<spec>/response_check.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import SPEC, SPLITS, Features, hour_of      # noqa: E402
from wedge.gnn.marginal import load_models                          # noqa: E402

NODES = ["GB", "NL", "BE", "RS", "HU"]
W, DELTA = 24, 100.0


def graph_slopes(models, fe, n, hours, dev):
    src, dst = torch.tensor(fe.src, device=dev), torch.tensor(fe.dst, device=dev)
    mask = torch.tensor(fe.disp_mask, device=dev)
    ef = torch.tensor(fe.ef_disp, device=dev)
    out = []
    for m in models:
        m.eval()
        sl = []
        with torch.no_grad():
            for i in range(0, len(hours), 256):
                hs = hours[i:i + 256]
                xs, mos, tas, ccs = (torch.tensor(np.stack(a), device=dev)
                                     for a in zip(*[fe.sample(int(t), W) for t in hs]))
                r = torch.tensor(fe.resid[:, hs].T, device=dev)
                g0, _, _ = m(xs, mos, tas, ccs, src, dst, r, mask)
                xp, rp = xs.clone(), r.clone()
                xp[:, -1, n, 3] += DELTA / 1000.0 / float(fe.rs[n, 0])
                rp[:, n] += DELTA / 1000.0
                g1, _, _ = m(xp, mos, tas, ccs, src, dst, rp, mask)
                sl.append((((g1 - g0)[:, n] * ef).sum(-1) * 1000.0 / DELTA).cpu().numpy())
        out.append(np.concatenate(sl))
    return np.mean(out, 0)


def tree_slopes(fe, n, hours):
    import xgboost as xgb
    tr = np.arange(W, hour_of(SPLITS["train"][1]), 3)
    ok = fe.disp_obs_ok[n, tr]
    Xtr = fe.X[tr, n][ok]
    tot = fe.resid[n, tr][ok]
    X0 = fe.X[hours, n].copy()
    X1 = X0.copy()
    X1[:, 3] += DELTA / 1000.0 / float(fe.rs[n, 0])
    R0 = fe.resid[n, hours] * 1000
    R1 = R0 + DELTA
    e0, e1 = np.zeros(len(hours)), np.zeros(len(hours))
    for k in range(fe.gen_disp.shape[1]):
        y = np.where(tot > 0, fe.gen_disp[n, k, tr][ok] / np.maximum(tot, 1e-6), 0)
        if y.max() <= 0:
            continue
        mdl = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, tree_method="hist",
                               device="cuda", random_state=0)
        mdl.fit(Xtr, y)
        e0 += np.clip(mdl.predict(X0), 0, 1) * fe.ef_disp[k] * R0
        e1 += np.clip(mdl.predict(X1), 0, 1) * fe.ef_disp[k] * R1
    return (e1 - e0) / DELTA


def season(hours):
    doy = (hours // 24) % 365
    return np.minimum(doy // 92, 3).astype(int)


def hourly_mef(fe, n, E, R, hours):
    tr = np.arange(W, hour_of(SPLITS["train"][1]))
    ok = fe.disp_obs_ok[n, tr] & fe.disp_obs_ok[n, tr - 1]
    t = tr[ok]
    dE, dR = E[t] - E[t - 1], R[t] - R[t - 1]
    cell = (t % 24) * 4 + season(t)
    slope = np.full(96, np.nan)
    for c in range(96):
        s = cell == c
        if s.sum() > 50 and np.var(dR[s]) > 0:
            slope[c] = np.polyfit(dR[s], dE[s], 1)[0]
    glob_s = np.polyfit(dR, dE, 1)[0]
    slope = np.where(np.isfinite(slope), slope, glob_s)
    return slope[(hours % 24) * 4 + season(hours)]


def ols(x, y):
    X = np.stack([np.ones_like(x), x], 1)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    cov = np.linalg.inv(X.T @ X) * (res @ res) / (len(y) - 2)
    return float(b[1]), float(np.sqrt(cov[1, 1]))


def evaluate(mbar, dE, dR):
    pred = mbar * dR
    beta, se = ols(pred, dE)
    r2 = 1 - ((dE - pred) ** 2).sum() / ((dE - dE.mean()) ** 2).sum()
    q = np.quantile(mbar, [0, .2, .4, .6, .8, 1])
    bins = []
    for j in range(5):
        s = (mbar >= q[j]) & (mbar <= q[j + 1])
        if s.sum() > 30:
            b, sb = ols(dR[s], dE[s])
            bins.append(dict(model=float(mbar[s].mean()), empirical=b, se=sb, n=int(s.sum())))
    mdl = np.array([b["model"] for b in bins])
    emp = np.array([b["empirical"] for b in bins])
    return dict(calibration_slope=beta, calibration_se=se, r2_oos=float(r2),
                bin_mae=float(np.abs(mdl - emp).mean()), bins=bins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", default="y2025,charged")
    a = ap.parse_args()
    dev = "cuda"
    fe = Features()
    idx = {n: i for i, n in enumerate(fe.nodes)}
    graph, nograph = load_models(fe, dev, ""), load_models(fe, dev, "dropTACC")
    E_all = (fe.gen_disp * fe.ef_disp[None, :, None]).sum(1) * 1000          # (N, H) t/h
    R_all = fe.resid * 1000                                                  # MW
    res = {}
    for name in NODES:
        n = idx[name]
        hours = np.concatenate([np.arange(*(hour_of(v) for v in SPLITS[p])) for p in a.periods.split(",")])
        hours = hours[fe.disp_obs_ok[n, hours] & fe.disp_obs_ok[n, hours - 1]]
        both = np.unique(np.concatenate([hours, hours - 1]))
        pos = {h: i for i, h in enumerate(both)}
        i1, i0 = np.array([pos[h] for h in hours]), np.array([pos[h - 1] for h in hours])
        E, R = E_all[n], R_all[n]
        dE, dR = E[hours] - E[hours - 1], R[hours] - R[hours - 1]
        keep = np.abs(dR) > 1.0
        slopes = {}
        if graph:
            s = graph_slopes(graph, fe, n, both, dev)
            slopes["graph"] = 0.5 * (s[i1] + s[i0])
        if nograph:
            s = graph_slopes(nograph, fe, n, both, dev)
            slopes["nograph"] = 0.5 * (s[i1] + s[i0])
        s = tree_slopes(fe, n, both)
        slopes["trees"] = 0.5 * (s[i1] + s[i0])
        s = hourly_mef(fe, n, E, R, both)
        slopes["hourly_mef"] = 0.5 * (s[i1] + s[i0])
        with np.errstate(all="ignore"):
            s = np.nan_to_num(E[both] / np.maximum(R[both], 100.0))       # floor avoids near-zero residuals
        slopes["average"] = 0.5 * (s[i1] + s[i0])
        res[name] = {"n_hours": int(keep.sum()), "sd_dE": float(dE[keep].std())}
        for k, v in slopes.items():
            res[name][k] = evaluate(v[keep], dE[keep], dR[keep])
            res[name][k]["mean_slope"] = float(v[keep].mean())
        print(name, {k: (round(res[name][k]["calibration_slope"], 2), round(res[name][k]["r2_oos"], 3),
                         round(res[name][k]["bin_mae"], 3)) for k in slopes}, flush=True)
    out = pathlib.Path(f"data/processed/gnn/{SPEC}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "response_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
