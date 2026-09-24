"""Within-window placebo for the outage design (A52).

The event fixed-effect slope of dY on the lost flow L identifies the outage response only if, in normal operation,
the within-window slope of the counterfactual residual dY on Fhat is zero. This script estimates that slope on
pseudo-events: non-overlapping windows in normal operation with the length distribution of the real events, and
reports the difference-in-slopes estimator  beta_outage_FE - beta_pseudo_FE  with a bootstrap over events and
pseudo-events. Output: data/processed/gnn/outage_did.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import DSFX, Features, hour_of  # noqa: E402
from wedge.gnn.outage_study import COMMISSIONED, LINKS, fe_ols, merge, oof_predict, spells   # noqa: E402

KEYS = ["R_m", "R_x", "NIother_m", "NIother_x"] + [f"E_{n}" for n in ("DE", "NL", "GB", "BE", "FR", "PL", "ES", "IT", "CZ",
                                                                        "AT", "DK", "NO")] + ["R_DE", "R_NO", "R_FR"]


def main(late=False):
    fe = Features()
    d = np.load(f"data/processed/gnn/dataset{DSFX}.npz")
    ci = {c: i for i, c in enumerate(fe.meta["classes"])}
    gen, load = np.nan_to_num(d["gen"]), np.nan_to_num(d["load"])
    H, idx, t = fe.H, {n: i for i, n in enumerate(fe.nodes)}, np.arange(fe.H)
    X = np.column_stack([a.T for a in [load, gen[:, ci["wind"]], gen[:, ci["solar"]], fe.z_nd]]
                        + [np.nan_to_num(d["glob"]), np.sin(2 * np.pi * (t % 24) / 24), np.cos(2 * np.pi * (t % 24) / 24),
                           (t // 24) % 7, np.sin(2 * np.pi * t / 8766), np.cos(2 * np.pi * t / 8766), t / 8766.0])
    E = (fe.gen_disp * fe.ef_disp[None, :, None]).sum(1) * 1000
    R, NI = fe.resid * 1000, fe.netimp * 1000
    out = {}
    rng = np.random.default_rng(7)
    for b in ("GB->NL", "GB->BE"):
        x, m = LINKS[b]
        xi, mi = idx[x], idx[m]
        F = fe.F[xi, mi]
        fin = np.isfinite(F)
        zero = fin & (np.abs(F) < 1.0)
        start = hour_of(COMMISSIONED[b]) if b in COMMISSIONED else 0
        ev = merge([(s, e) for s, e in spells(zero) if s >= start])
        excl = np.zeros(H, bool)
        for s, e in ev:
            excl[max(s - 48, 0):e + 24] = True
        if late:                                   # held-out validation events: start in 2023 or later
            ev = [(s, e) for s, e in ev if s >= hour_of("2023-01-01")]
        for s, e in ev:
            excl[max(s - 48, 0):e + 24] = True
        excl[:start] = True
        normal = fin & ~zero & ~excl & (load[xi] > 0) & (load[mi] > 0)
        Fhat = oof_predict(X, np.nan_to_num(F), normal, t // 168)
        out_h, evid = np.zeros(H, bool), np.full(H, -1)
        for n_, (s, e) in enumerate(ev):
            out_h[s:e] = True
            evid[s:e] = n_
        # pseudo-events: 5 x as many windows as events, lengths drawn from the event lengths, fully normal
        lens = [e - s for s, e in ev]
        pid, taken, n_ps = np.full(H, -1), np.zeros(H, bool), 0
        tries = 0
        while n_ps < 5 * len(ev) and tries < 100000:
            tries += 1
            L_ = int(rng.choice(lens))
            s = int(rng.integers(start + 48, H - L_ - 1))
            if normal[s:s + L_].all() and not taken[s:s + L_].any():
                pid[s:s + L_] = n_ps
                taken[s:s + L_] = True
                n_ps += 1
        outcomes = {"R_m": R[mi], "R_x": R[xi], "NIother_m": NI[mi] - np.nan_to_num(F), "NIother_x": NI[xi] + np.nan_to_num(F)}
        for n in fe.nodes:
            outcomes[f"E_{n}"] = E[idx[n]]
            outcomes[f"R_{n}"] = R[idx[n]]
        obs_ok = fe.disp_obs_ok[mi] & fe.disp_obs_ok[xi]
        r = {"events": len(ev), "pseudo_events": n_ps}
        for k in KEYS:
            Y = outcomes[k]
            okY = obs_ok & np.isfinite(Y)
            if k[:2] in ("E_", "R_") and k not in ("R_m", "R_x"):
                okY &= fe.disp_obs_ok[idx[k.split("_", 1)[1]]]
            Yhat = oof_predict(X, np.nan_to_num(Y), okY & normal & (pid < 0), t // 168)
            dY = Y - Yhat
            so, sp = out_h & okY, (pid >= 0) & okY
            bo, se_o = fe_ols(Fhat[so], dY[so], evid[so])
            bp, se_p = fe_ols(Fhat[sp], dY[sp], pid[sp])
            # bootstrap over events and pseudo-events for the difference
            ev_ids, ps_ids = np.unique(evid[so]), np.unique(pid[sp])
            diffs = []
            for _ in range(300):
                e_s = rng.choice(ev_ids, len(ev_ids))
                p_s = rng.choice(ps_ids, len(ps_ids))
                mo = np.concatenate([np.where(so & (evid == c))[0] for c in e_s])
                gm = np.concatenate([np.full((so & (evid == c)).sum(), j) for j, c in enumerate(e_s)])
                mp = np.concatenate([np.where(sp & (pid == c))[0] for c in p_s])
                gp = np.concatenate([np.full((sp & (pid == c)).sum(), j) for j, c in enumerate(p_s)])
                diffs.append(fe_ols(Fhat[mo], dY[mo], gm)[0] - fe_ols(Fhat[mp], dY[mp], gp)[0])
            diffs = np.array(diffs)
            r[k] = {"outage_fe": bo, "se_outage_fe": se_o, "pseudo_fe": bp, "se_pseudo_fe": se_p,
                    "did": bo - bp, "did_se_boot": float(np.nanstd(diffs)),
                    "did_ci95": [float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))]}
            print(b, k, "outage %.2f (%.2f) pseudo %.2f (%.2f) did %.2f [%.2f, %.2f]" % (
                bo, se_o, bp, se_p, bo - bp, *r[k]["did_ci95"]), flush=True)
        out[b] = r
    pathlib.Path("data/processed/gnn/outage_did" + ("_late" if late else "") + DSFX + ".json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(late="--late" in sys.argv)
