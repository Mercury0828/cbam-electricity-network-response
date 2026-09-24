"""Interconnector outages as link-specific natural experiments for network redistribution (A51).

A forced outage removes one link while leaving every other link available, which is the intervention the CBAM charge
approximates (a cost wedge on one link), unlike exporter-wide shocks such as exporter wind.

Events: spells of at least 24 hours with zero flow on the link (|F| < 1 MW) while the link normally trades.
Counterfactual: for every outcome Y (flow on the link, residual demand R_i, dispatchable emissions E_i and net
imports of each zone) a gradient-boosted model predicts Y from exogenous drivers only: load, wind, solar and
non-dispatchable output of all 17 zones, fuel and carbon prices, calendar terms and a time trend. Predictions are
out-of-fold by calendar week, so outage and non-outage hours are predicted alike and the model never sees the outage.
The lost flow is L_t = Fhat_t (what the link would have carried); the effect on Y is dY_t = Y_t - Yhat_t.
Estimands (OLS of dY on L over outage hours, standard errors clustered by day):
  a_m    = -d R_m / L          share of the lost import that the importer's own plants replace
  a_x    = -d R_x / (-L)       share of the lost export that the exporter's own plants no longer produce
  e_i    = d E_i / L           emission change in zone i per MWh of lost import (network emission response)
Placebo: the same regression over non-outage hours (L = Fhat, no outage) should give zero.
Output: data/processed/gnn/outage_study.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import DSFX, Features, hour_of  # noqa: E402

LINKS = {"GB->NL": ("GB", "NL"), "GB->BE": ("GB", "BE"), "RS->HU": ("RS", "HU")}
COMMISSIONED = {"GB->BE": "2019-02-01"}          # Nemo Link commercial operation 31 Jan 2019
MERGE_GAP = 72                                    # spells separated by less than this many hours form one event


def merge(ev, gap=MERGE_GAP):
    out = []
    for s, e in ev:
        if out and s - out[-1][1] < gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def spells(zero, min_len=24):
    out, s = [], None
    for t, v in enumerate(zero):
        if v and s is None:
            s = t
        if not v and s is not None:
            if t - s >= min_len:
                out.append((s, t))
            s = None
    return out


def oof_predict(X, y, ok, week, dev="cpu"):
    import xgboost as xgb
    pred = np.full(len(y), np.nan)
    folds = week % 5
    for k in range(5):
        tr = ok & (folds != k)
        te = folds == k
        m = xgb.XGBRegressor(n_estimators=400, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", device=dev, random_state=k)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def cluster_ols(x, y, g):
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    meat = np.zeros((2, 2))
    for c in np.unique(g):
        s = g == c
        u = X[s].T @ e[s]
        meat += np.outer(u, u)
    V = XtXi @ meat @ XtXi
    return float(b[1]), float(np.sqrt(V[1, 1]))


def fe_ols(x, y, g):
    """OLS slope with group fixed effects (within-group variation only); standard error clustered by group."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], g[ok]
    xd, yd = x.copy(), y.copy()
    for c in np.unique(g):
        s = g == c
        xd[s] -= x[s].mean()
        yd[s] -= y[s].mean()
    den = xd @ xd
    if den <= 0:
        return np.nan, np.nan
    b = (xd @ yd) / den
    e = yd - b * xd
    meat = sum(((xd[g == c] * e[g == c]).sum()) ** 2 for c in np.unique(g))
    return float(b), float(np.sqrt(meat) / den)


def main():
    fe = Features()
    d = np.load(f"data/processed/gnn/dataset{DSFX}.npz")
    ci = {c: i for i, c in enumerate(fe.meta["classes"])}
    gen, load = np.nan_to_num(d["gen"]), np.nan_to_num(d["load"])
    H, N = fe.H, fe.N
    idx = {n: i for i, n in enumerate(fe.nodes)}
    t = np.arange(H)
    exo = [load, gen[:, ci["wind"]], gen[:, ci["solar"]], fe.z_nd]
    X = np.column_stack([a.T for a in exo] + [np.nan_to_num(d["glob"]),
                        np.sin(2 * np.pi * (t % 24) / 24), np.cos(2 * np.pi * (t % 24) / 24),
                        (t // 24) % 7, np.sin(2 * np.pi * t / 8766), np.cos(2 * np.pi * t / 8766), t / 8766.0])
    week = t // 168
    E = (fe.gen_disp * fe.ef_disp[None, :, None]).sum(1) * 1000                       # t/h
    R = fe.resid * 1000                                                                # MW
    NI = fe.netimp * 1000
    res = {}
    for b, (x, m) in LINKS.items():
        xi, mi = idx[x], idx[m]
        F = fe.F[xi, mi]
        fin = np.isfinite(F)
        zero = fin & (np.abs(F) < 1.0)
        start = hour_of(COMMISSIONED[b]) if b in COMMISSIONED else 0
        ev = merge([(s, e) for s, e in spells(zero) if s >= start])
        excl = np.zeros(H, bool)
        for s, e in ev:
            excl[s:e + 24] = True                  # outage hours plus the day after restoration (not placebo)
        excl[:start] = True
        out_h = np.zeros(H, bool)
        for s, e in ev:
            out_h[s:e] = True
        base_ok = fin & ~zero & ~excl & (load[xi] > 0) & (load[mi] > 0)
        Fhat = oof_predict(X, np.nan_to_num(F), base_ok, week)
        # outcomes
        outcomes = {"R_m": R[mi], "R_x": R[xi], "NIother_m": NI[mi] - np.nan_to_num(F),
                    "NIother_x": NI[xi] + np.nan_to_num(F)}
        for j in range(N):
            outcomes[f"E_{fe.nodes[j]}"] = E[j]
            outcomes[f"R_{fe.nodes[j]}"] = R[j]
        obs_ok = fe.disp_obs_ok[mi] & fe.disp_obs_ok[xi]
        r = {"events": len(ev), "outage_hours": int(out_h.sum()),
             "mean_lost_flow_mw": float(Fhat[out_h].mean()), "mean_abs_lost_flow_mw": float(np.abs(Fhat[out_h]).mean())}
        day = t // 24
        evid = np.full(H, -1)
        evid_pre = np.full(H, -1)
        pre = np.zeros(H, bool)
        for n_, (s, e) in enumerate(ev):
            evid[s:e] = n_
            pre[max(s - 48, 0):s] = True
            evid_pre[max(s - 48, 0):s] = n_
        pre &= ~excl & fin & (np.abs(np.nan_to_num(F)) > 50)
        early = t < hour_of("2023-01-01")
        keep = {"hours": np.where(out_h)[0], "Fhat": Fhat[out_h], "evid": evid[out_h]}
        for k, Y in outcomes.items():
            okY = obs_ok & np.isfinite(Y)
            if k.startswith(("E_", "R_")) and k not in ("R_m", "R_x"):
                j = idx[k.split("_", 1)[1]]
                okY = okY & fe.disp_obs_ok[j]
            Yhat = oof_predict(X, np.nan_to_num(Y), okY & ~excl, week)
            dY = Y - Yhat
            so = out_h & okY
            sp = ~out_h & okY & base_ok
            bo, so_se = cluster_ols(Fhat[so], dY[so], day[so])
            rng = np.random.default_rng(0)
            pl = rng.choice(np.where(sp)[0], min(20000, sp.sum()), replace=False)
            bp, sp_se = cluster_ols(Fhat[pl], dY[pl], day[pl])
            be, be_se = cluster_ols(Fhat[so], dY[so], evid[so])                      # clustered by event
            spre = pre & okY & base_ok
            bpre, bpre_se = cluster_ols(Fhat[spre], dY[spre], day[spre]) if spre.sum() > 50 else (np.nan, np.nan)
            b1, s1 = cluster_ols(Fhat[so & early], dY[so & early], evid[so & early]) if (so & early).sum() > 50                 else (np.nan, np.nan)
            b2, s2 = cluster_ols(Fhat[so & ~early], dY[so & ~early], evid[so & ~early]) if (so & ~early).sum() > 50                 else (np.nan, np.nan)
            brex = (t >= hour_of("2020-12-01")) & (t < hour_of("2021-07-01"))
            bx, sx = cluster_ols(Fhat[so & ~brex], dY[so & ~brex], evid[so & ~brex]) if (so & ~brex).sum() > 50                 else (np.nan, np.nan)
            bfe, sfe = fe_ols(Fhat[so], dY[so], evid[so])
            bfp, sfp = fe_ols(Fhat[spre], dY[spre], evid_pre[spre]) if spre.sum() > 50 else (np.nan, np.nan)
            r[k] = {"event_fe": bfe, "se_event_fe": sfe, "pre48_event_fe": bfp, "se_pre48_event_fe": sfp,
                    "excl_2020Dec_2021Jun": bx, "se_excl": sx, "outage": bo, "se": so_se, "se_event": be_se, "placebo": bp, "placebo_se": sp_se,
                    "pre48": bpre, "pre48_se": bpre_se, "2019_2022": b1, "se_2019_2022": s1,
                    "2023_2026": b2, "se_2023_2026": s2}
            keep[k] = dY[out_h]
            print(b, k, round(bo, 3), round(so_se, 3), "| placebo", round(bp, 3), round(sp_se, 3), flush=True)
        # absorption shares and network emission response per MWh of lost import x -> m
        r["absorption_m"] = r["R_m"]["outage"]            # dR_m per MW of lost import (1 = one for one)
        r["absorption_x"] = -r["R_x"]["outage"]           # -dR_x per MW of lost export (1 = one for one)
        r["network_emission_per_mwh"] = float(sum(r[f"E_{n}"]["outage"] for n in fe.nodes))
        # balance audit: dR_m + dNIother_m - 1 and dR_x + dNIother_x + 1 should be close to zero
        r["balance_residual_m"] = r["R_m"]["outage"] + r["NIother_m"]["outage"] - 1.0
        r["balance_residual_x"] = r["R_x"]["outage"] + r["NIother_x"]["outage"] + 1.0
        r["n_events_2019_2022"] = int(len({evid[h] for h in np.where(out_h & early)[0]}))
        r["n_events_2023_2026"] = int(len({evid[h] for h in np.where(out_h & ~early)[0]}))
        np.savez_compressed(f"data/processed/gnn/outage_{b.replace('->', '_')}{DSFX}.npz", **keep)
        res[b] = r
        print(b, {k: r[k] for k in ("events", "outage_hours", "mean_lost_flow_mw", "absorption_m", "absorption_x",
                                    "network_emission_per_mwh")}, flush=True)
    pathlib.Path(f"data/processed/gnn/outage_study{DSFX}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
