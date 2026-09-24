"""How much of an exogenous change in a charged border flow does the importer's own dispatch absorb? (A51)

The flow counterfactual of marginal.py assigns the whole change in delivered import to the importer's residual
(R_m += delta), holding every other interconnector fixed. In a coupled market the importer can instead draw more on
its other neighbours. This script measures the absorption share empirically.

For border x -> m and hour-to-hour changes, 2019-2026:
  first stage   dF_xm  on  d(exporter wind) , d(exporter load)       (exporter-side shocks)
  second stage  dR_m   on  dF_xm_hat,  controls d(importer load), d(importer wind+solar), hour-of-day, month
  also          d(importer net import from all other links) on dF_xm_hat
A coefficient of -1 on dR_m means that the importer's own plants absorb the whole change (the one-for-one
assumption); a coefficient near 0 means that the change is passed on to other links and zones.
Output: data/processed/gnn/redistribution_iv.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import DSFX, Features  # noqa: E402

BORDERS = {"GB->NL": ("GB", "NL"), "GB->BE": ("GB", "BE"), "RS->HU": ("RS", "HU")}


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def iv(y, x, Z, W):
    """2SLS of y on x (endogenous) with instruments Z and exogenous controls W; HC0 standard error."""
    ZW = np.column_stack([Z, W])
    xh = ZW @ ols(ZW, x)
    X2 = np.column_stack([xh, W])
    b = ols(X2, y)
    res = y - np.column_stack([x, W]) @ b
    XtXi = np.linalg.pinv(X2.T @ X2)
    V = XtXi @ (X2.T * res ** 2) @ X2 @ XtXi
    f_first = None
    r_full = x - ZW @ ols(ZW, x)
    r_red = x - W @ ols(W, x)
    q, n, k = Z.shape[1], len(x), ZW.shape[1]
    f_first = float(((r_red @ r_red - r_full @ r_full) / q) / ((r_full @ r_full) / (n - k)))
    return float(b[0]), float(np.sqrt(V[0, 0])), f_first


def main():
    fe = Features()
    d = np.load(f"data/processed/gnn/dataset{DSFX}.npz")
    classes = fe.meta["classes"]
    ci = {c: i for i, c in enumerate(classes)}
    gen, load = np.nan_to_num(d["gen"]), np.nan_to_num(d["load"])
    wind = gen[:, ci["wind"]]
    vre = wind + gen[:, ci["solar"]]
    idx = {n: i for i, n in enumerate(fe.nodes)}
    H = fe.H
    hod = np.arange(H) % 24
    out = {}
    for b, (x, m) in BORDERS.items():
        xi, mi = idx[x], idx[m]
        F = fe.F[xi, mi]
        R = fe.resid[mi] * 1000
        other = fe.netimp[mi] * 1000 - F                                 # net import of m from all other links
        t = np.arange(1, H)
        ok = fe.disp_obs_ok[mi, t] & fe.disp_obs_ok[mi, t - 1] & (load[xi, t] > 0) & (load[mi, t] > 0)
        t = t[ok]
        dF, dR, dO = F[t] - F[t - 1], R[t] - R[t - 1], other[t] - other[t - 1]
        Z = np.column_stack([wind[xi, t] - wind[xi, t - 1], load[xi, t] - load[xi, t - 1]])
        Wc = np.column_stack([np.ones(len(t)), load[mi, t] - load[mi, t - 1], vre[mi, t] - vre[mi, t - 1]]
                             + [(hod[t] == k).astype(float) for k in range(1, 24)])
        bR, sR, fR = iv(dR, dF, Z, Wc)
        bO, sO, _ = iv(dO, dF, Z, Wc)
        ols_R = float(ols(np.column_stack([dF, Wc]), dR)[0])
        out[b] = {"n": int(len(t)), "iv_dR_on_dF": bR, "se": sR, "first_stage_F": fR,
                  "iv_dOtherImport_on_dF": bO, "se_other": sO, "ols_dR_on_dF": ols_R}
        print(b, {k: round(v, 3) for k, v in out[b].items()}, flush=True)
    pathlib.Path("data/processed/gnn/redistribution_iv.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
