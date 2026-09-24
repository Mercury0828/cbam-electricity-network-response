"""Synthetic full-pipeline check of the outage estimator.

Synthetic outcomes with a known response to the flow of the focal link are built on the real hours, drivers, flows,
outage events, treated blocks and pseudo windows of outage_v2.py, and pass through the identical estimator
(outage_v2.design / estimate: out-of-fold nuisance, block and block x hour fixed effects, direction split, pseudo-window
subtraction, event bootstrap). Outcome model for outcome k of link x -> m in hour t (signed flow F_t, zero in outage
hours):
    Y_t = g_k(X_t) - beta_k(t) * F_t + e_t
so removing a would-be flow F* changes Y by +beta_k(t) F*, and beta_k is the slope the estimator should recover.
g_k is an xgboost fit of the real outcome k on the drivers over the nuisance training hours; e is a weekly block
bootstrap (donor weeks from the same calendar quarter) of the real out-of-fold residuals of outcome k, which keeps
their scale, autocorrelation and daily pattern. Hours in which the real outcome is missing stay missing.
Truths (importer plants, importer other links, exporter plants, exporter other links):
    local       ( 1.0, 0.0, -1.0,  0.0)   two-zone local replacement
    rerouting   ( 0.2, 0.8, -0.2, -0.8)
    placebo     ( 0.0, 0.0,  0.0,  0.0)
    state       ( a_m(t), 1 - a_m(t), -a_x(t), -(1 - a_x(t)) ),  a_z(t) = 0.3 + 0.6 * load rank of zone z in hour t
The target of the state truth is computed with the model-side construction of outage_model_compare (within-block
regression of the true effects beta(t) L_t on the dose over treated hours, zero effect in pseudo windows), so the check
also covers how the model and the two-zone baseline enter the comparison.
Also reported: the construction in which normal windows use the dose Fhat - F instead of the would-be flow Fhat (which
would difference a true response away), and a device check (real outcomes, GPU nuisance vs the frozen CPU run).
Output: data/processed/gnn/outage_synthetic{SYNTH_TAG}{DSFX}.json (SYNTH_REPS noise draws, default 3)
Usage: WEDGE_GNN_SPEC=v4 [SYNTH_REPS=10 SYNTH_TAG=10] python src/wedge/gnn/outage_synthetic.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import DSFX, Features                                              # noqa: E402
from wedge.gnn.outage_model_compare import model_slopes                                   # noqa: E402
from wedge.gnn.outage_v2 import FOCAL, design, drivers, estimate, oof, outcome_ok, year_q  # noqa: E402

DEV = "cuda"                        # frozen run: CPU; same estimator settings (see device_check in the output)
REPS = int(os.environ.get("SYNTH_REPS", "3"))
STAG = os.environ.get("SYNTH_TAG", "")
KEYS = ("R_m", "NIother_m", "R_x", "NIother_x")
TRUTHS = {"local": (1.0, 0.0, -1.0, 0.0), "rerouting": (0.2, 0.8, -0.2, -0.8), "placebo": (0.0, 0.0, 0.0, 0.0),
          "state": None}
FES = ("A_block", "B_block_hour")
COMP = ("plus", "minus", "pooled")


def fit_in_sample(X, y, ok):
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=400, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         tree_method="hist", device=DEV, random_state=99, n_jobs=16)
    m.fit(X[ok], y[ok])
    return m.predict(X)


def load_rank(v):
    """Percentile rank in [0, 1] of each hour's load within the sample."""
    r = np.argsort(np.argsort(v))
    return r / max(len(v) - 1, 1)


def week_bootstrap(res, ok, H, rng):
    """Weekly block bootstrap of residuals: each week receives a donor week of the same calendar quarter in which at
    least 95% of the hours carry a valid residual; the donor's remaining hours take its nearest valid hour."""
    nw = (H + 167) // 168
    q = np.array([year_q(w * 168)[1] for w in range(nw)])
    share = np.array([ok[w * 168:(w + 1) * 168].mean() if (w + 1) * 168 <= H else 0.0 for w in range(nw)])
    e = np.zeros(H)
    for w in range(nw):
        pool = np.where((share >= 0.95) & (q == q[w]))[0]
        dw = rng.choice(pool)
        seg, v = res[dw * 168:(dw + 1) * 168], ok[dw * 168:(dw + 1) * 168]
        good = np.where(v)[0]
        fill = good[np.abs(np.arange(168)[:, None] - good[None, :]).argmin(1)]
        n = min(168, H - w * 168)
        e[w * 168:w * 168 + n] = seg[fill][:n]
    return e


def near_events(DS, H):
    """Hours within 48 h before to 24 h after any zero-flow event of either focal link (their residuals carry the
    real outage response and are not used as synthetic noise)."""
    m = np.zeros(H, bool)
    for D in DS.values():
        for s, e in D["ev"]:
            m[max(s - 48, 0):min(e + 24, H)] = True
    return m


def main():
    t0 = time.time()
    fe = Features()
    d = np.load(f"data/processed/gnn/dataset{DSFX}.npz")
    idx = {n: i for i, n in enumerate(fe.nodes)}
    X_full, _, week, load, _, R, NI = drivers(fe, d)
    DS = design(fe, d, np.random.default_rng(11))                     # identical structure and draws
    away = ~near_events(DS, fe.H)
    frozen = json.loads(pathlib.Path(f"data/processed/gnn/outage_v2{DSFX}.json").read_text(encoding="utf-8"))
    out = {"_meta": {"reps": REPS, "device": DEV, "truths": {k: v for k, v in TRUTHS.items() if v},
                     "state_truth": "a_z(t) = 0.3 + 0.6 * load rank of zone z (importer for _m, exporter for _x)",
                     "order": list(KEYS), "components": list(COMP)}}
    for li, (b, D) in enumerate(DS.items()):
        H, F = fe.H, np.nan_to_num(D["F"])
        mi, xi, so, sp, train = D["mi"], D["xi"], D["so_all"], D["sp_all"], D["train"]
        z = np.load(f"data/processed/gnn/outage_v2_blocks_{b.replace('->', '_')}{DSFX}.npz")
        Luse, Fh = np.full(H, np.nan), np.full(H, np.nan)
        Luse[so], Fh[sp] = z["L"], z["p_L"]                            # frozen dose and would-be flow
        Fh_alt = Fh.copy()
        Fh_alt[sp] = z["p_L"] - F[sp]                                  # alternative: dose Fhat - F in normal windows
        hours, blk, L = z["hours"], z["block"], z["L"]
        real = {"R_m": R[mi], "NIother_m": NI[mi] - F, "R_x": R[xi], "NIother_x": NI[xi] + F}
        a_m, a_x = 0.3 + 0.6 * load_rank(load[mi]), 0.3 + 0.6 * load_rank(load[xi])
        state = {"R_m": a_m, "NIother_m": 1.0 - a_m, "R_x": -a_x, "NIother_x": -(1.0 - a_x)}
        base, dev_check = {}, {}
        for k, Y in real.items():
            okY = outcome_ok(fe, idx, k, D, Y)
            fit = okY & train
            g = fit_in_sample(X_full, np.nan_to_num(Y), fit)
            res = np.nan_to_num(Y - oof(X_full, np.nan_to_num(Y), fit, week, DEV))
            base[k] = (g, res, okY, okY & away, np.isfinite(Y))
            chk, _ = estimate(res, okY, D, Luse, Fh)                   # real outcome, GPU nuisance
            dev_check[k] = {fes: {"did_gpu": chk[fes]["did"], "did_frozen_cpu": frozen[b]["outcomes"][f"{k}|full"][fes]["did"]}
                            for fes in FES}
            print(b, k, "device check A did gpu", np.round(chk["A_block"]["did"], 3), "frozen",
                  np.round(frozen[b]["outcomes"][f"{k}|full"]["A_block"]["did"], 3), f"{time.time() - t0:.0f}s", flush=True)
        # targets: constant truths exactly; state truth with the model-side construction
        targets = {}
        for tname, tv in TRUTHS.items():
            targets[tname] = {}
            for j, k in enumerate(KEYS):
                okk = base[k][2][hours]
                if tv is not None:
                    targets[tname][k] = {fes: [tv[j]] * 3 for fes in FES}
                else:
                    eff = state[k][hours] * L
                    targets[tname][k] = {fes: model_slopes(hours[okk], blk[okk], L[okk], eff[okk], fes).tolist()
                                         for fes in FES}
        runs = {t: [] for t in TRUTHS}
        alt = []
        for rep in range(REPS):
            for tname, tv in TRUTHS.items():
                rec, rec_alt = {}, {}
                for j, k in enumerate(KEYS):
                    g, res, okY, fit, fin = base[k]
                    rng = np.random.default_rng(1000 * (li + 1) + 10 * rep + j)
                    e = week_bootstrap(res, fit, H, rng)
                    beta = state[k] if tv is None else tv[j]
                    Ys = g - beta * F + e
                    Ys[~fin] = np.nan
                    Yhat = oof(X_full, np.nan_to_num(Ys), okY & train, week, DEV)
                    dY = np.nan_to_num(Ys - Yhat)
                    o_k, _ = estimate(dY, okY, D, Luse, Fh)
                    rec[k] = {fes: {kk: o_k[fes][kk] for kk in ("outage", "pseudo", "did", "ci95")} for fes in FES}
                    if tname == "local":
                        a_k, _ = estimate(dY, okY, D, Luse, Fh_alt)
                        rec_alt[k] = {fes: {"pseudo": a_k[fes]["pseudo"], "did": a_k[fes]["did"]} for fes in FES}
                runs[tname].append(rec)
                if tname == "local":
                    alt.append(rec_alt)
                print(b, "rep", rep, tname, {k: np.round(rec[k]["A_block"]["did"], 2).tolist() for k in KEYS},
                      f"{time.time() - t0:.0f}s", flush=True)
        # summary: mean over replications, error against the target, interval coverage
        summ, cover, worst = {}, [], 0.0
        for tname in TRUTHS:
            summ[tname] = {}
            for k in KEYS:
                summ[tname][k] = {}
                for fes in FES:
                    tg = np.array(targets[tname][k][fes])
                    dd = np.array([r[k][fes]["did"] for r in runs[tname]])
                    oo = np.array([r[k][fes]["outage"] for r in runs[tname]])
                    pp = np.array([r[k][fes]["pseudo"] for r in runs[tname]])
                    ci = np.array([r[k][fes]["ci95"] for r in runs[tname]])          # (reps, 3, 2)
                    inside = (ci[:, :, 0] <= tg[None]) & (tg[None] <= ci[:, :, 1])
                    cover.extend(inside.ravel().tolist())
                    err = dd.mean(0) - tg
                    worst = max(worst, float(np.nanmax(np.abs(err))))
                    summ[tname][k][fes] = {"target": tg.tolist(), "did_mean": dd.mean(0).tolist(),
                                           "did_min": dd.min(0).tolist(), "did_max": dd.max(0).tolist(),
                                           "outage_mean": oo.mean(0).tolist(), "pseudo_mean": pp.mean(0).tolist(),
                                           "error_of_mean": err.tolist(), "ci_covers": inside.sum(0).tolist()}
        alt_summ = {k: {fes: {"did_mean": np.mean([a[k][fes]["did"] for a in alt], 0).tolist(),
                              "pseudo_mean": np.mean([a[k][fes]["pseudo"] for a in alt], 0).tolist()}
                        for fes in FES} for k in KEYS}
        out[b] = {"device_check": dev_check, "targets": targets, "summary": summ,
                  "alternative_normal_window_dose": alt_summ, "runs": runs,
                  "coverage_share": float(np.mean(cover)), "coverage_n": len(cover),
                  "max_abs_error_of_mean": worst, "treated_hours": int(so.sum()), "pseudo_hours": int(sp.sum())}
        print(b, "coverage", round(float(np.mean(cover)), 3), "max |error of mean|", round(worst, 3), flush=True)
    pathlib.Path(f"data/processed/gnn/outage_synthetic{STAG}{DSFX}.json").write_text(json.dumps(out, indent=1),
                                                                              encoding="utf-8")
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
