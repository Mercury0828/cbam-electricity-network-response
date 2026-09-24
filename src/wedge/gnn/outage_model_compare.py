"""Model-side outage responses with the IDENTICAL estimator as the empirical study (A54).

For the treated hours and blocks saved by outage_v2.py, each model supplies the effect of removing the dose L_t on the
link x -> m in hour t:
  graph / local   network-response model (netresp, tag r2, dataset v4): g_l = 0, dN_m += L_t, dN_x -= L_t / eta,
                  dR_i = model response, dE_i = m_i(t) dR_i with m_i the v4 dispatch head's local emission response
  one_for_one     dR_m = L_t, dR_x = -L_t / eta, all other zones 0 (two-zone accounting), dE via m_i
In pseudo windows no intervention exists, so the model effect is zero and the pseudo slope is zero. The model slope
is then the within-block slope of dR_i (dE_i) on [L+, L-] and on L, specifications A (block) and B (block x hour),
exactly as in outage_v2. Output: data/processed/gnn/outage_model_compare{DSFX}.json with per-outcome model slopes,
empirical DiD and 95% intervals, inside-interval flags and error summaries.
Usage: WEDGE_GNN_SPEC=v4 NETRESP_TAG=r2 python src/wedge/gnn/outage_model_compare.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import DSFX, Features                                      # noqa: E402
from wedge.gnn.netresp import ETA, TAG, Data, intervene, load_models, local_emission_slopes   # noqa: E402
RTAG = "" if TAG in ("", "r2") else f"_{TAG}"                             # A60: r3 must not overwrite r2
from wedge.gnn.outage_v2 import FOCAL, block_stats, slopes                         # noqa: E402
from wedge.gnn.outage_v2_post import outcome_mask                                  # noqa: E402

DEV = "cuda"


def model_slopes(hours, blk, L, dY, fes):
    hod = hours % 24
    g = blk if fes == "A_block" else blk * 24 + hod
    _, gi = np.unique(g, return_inverse=True)
    _, bi = np.unique(blk, return_inverse=True)
    S = block_stats(L, dY, bi, gi, bi.max() + 1)
    return slopes(S.sum(0))


def main():
    fe = Features()
    dat = Data(fe)
    idx = {n: i for i, n in enumerate(fe.nodes)}
    emp_all = json.loads(pathlib.Path(f"data/processed/gnn/outage_v2{DSFX}.json").read_text(encoding="utf-8"))
    out = {}
    for b, (x, m) in FOCAL.items():
        xi, mi = idx[x], idx[m]
        z = np.load(f"data/processed/gnn/outage_v2_blocks_{b.replace('->', '_')}{DSFX}.npz")
        hours, blk, L = z["hours"], z["block"], z["L"]
        # identical sample to the empirical side: no extra hour filter here; the outcome-specific
        # masks of outage_v2 are applied per outcome below. Hours with L == 0 carry no dose and are kept (effect 0).
        Ls = np.where(np.abs(L) > 1e-6, L, 1e-6)
        mslope = local_emission_slopes(fe, hours, range(fe.N), DEV)                 # (T, N)
        resp, respni = {}, {}
        for name, local in (("graph", False), ("local", True)):
            try:
                ms = load_models(fe, DEV, local)
            except FileNotFoundError:                                              # A60: r3 node-local may lag
                if name == "graph":
                    raise
                print("no node-local checkpoints, skipped", flush=True)
                continue
            rho, ni = intervene(ms, dat, hours, xi, mi, ETA[b], Ls, DEV, return_ni=True)
            resp[name] = rho * L[:, None]                                          # dR (T, N) MW
            respni[name] = ni * L[:, None]                                         # internal-link dNI (T, N) MW
        pos = L > 0                                                                # A56 (B1): losses on the sender
        dR1 = np.zeros((len(hours), fe.N))
        dR1[:, mi] = np.where(pos, L, L / ETA[b])
        dR1[:, xi] = np.where(pos, -L / ETA[b], -L)
        resp["one_for_one"] = dR1
        respni["one_for_one"] = np.zeros_like(dR1)                                 # own plants replace everything
        emp = emp_all[b]["outcomes"]
        res_b = {}
        for name, dR in resp.items():
            dE = dR * mslope
            rows = {}
            for key in list(emp):
                k, spec = key.split("|")
                if spec != "full":
                    continue
                if k == "R_m":
                    y = dR[:, mi]
                elif k == "R_x":
                    y = dR[:, xi]
                elif k == "NIother_m":
                    y = respni[name][:, mi]       # A56 (B7): solved internal-link flows (clamped link has g = 0)
                elif k == "NIother_x":
                    y = respni[name][:, xi]
                elif k.startswith("R_"):
                    y = dR[:, idx[k[2:]]]
                elif k.startswith("E_"):
                    y = dE[:, idx[k[2:]]]
                else:
                    continue
                rows[k] = {}
                okk = outcome_mask(fe, idx, k, x, m, hours)
                for fes in ("A_block", "B_block_hour"):
                    ms_ = model_slopes(hours[okk], blk[okk], L[okk], y[okk], fes)
                    e = emp[key][fes]
                    ci = np.array(e["ci95"])                                       # (3, 2): plus, minus, pooled
                    inside = [bool(ci[j, 0] <= ms_[j] <= ci[j, 1]) if np.all(np.isfinite(ci[j])) and np.isfinite(ms_[j])
                              else None for j in range(3)]
                    rows[k][fes] = {"model": ms_.tolist(), "empirical_did": e["did"], "ci95": e["ci95"],
                                    "inside": inside, "n_hours": int(okk.sum())}
            summ = {}
            for fes in ("A_block", "B_block_hour"):
                for j, lab in enumerate(("plus", "minus", "pooled")):
                    for grp, keys in (("R", [k for k in rows if k.startswith(("R_", "NIother"))
                                             and k not in (f"R_{x}", f"R_{m}")]),        # no endpoint aliases
                                      ("E", [k for k in rows if k.startswith("E_")])):
                        errs = [rows[k][fes]["model"][j] - rows[k][fes]["empirical_did"][j] for k in keys
                                if np.isfinite(rows[k][fes]["model"][j]) and np.isfinite(rows[k][fes]["empirical_did"][j])]
                        ins = [rows[k][fes]["inside"][j] for k in keys if rows[k][fes]["inside"][j] is not None]
                        summ[f"{fes}|{lab}|{grp}"] = {"rmse": float(np.sqrt(np.mean(np.square(errs)))) if errs else None,
                                                      "inside_share": float(np.mean(ins)) if ins else None,
                                                      "n": len(errs)}
            res_b[name] = {"rows": rows, "summary": summ}
            for fes in ("A_block", "B_block_hour"):
                s = summ
                print(b, name, fes, {lab: (round(s[f"{fes}|{lab}|R"]["rmse"] or np.nan, 3),
                                           round(s[f"{fes}|{lab}|R"]["inside_share"] or np.nan, 2),
                                           round(s[f"{fes}|{lab}|E"]["rmse"] or np.nan, 3),
                                           round(s[f"{fes}|{lab}|E"]["inside_share"] or np.nan, 2))
                                     for lab in ("plus", "minus", "pooled")}, flush=True)
        out[b] = res_b
    pathlib.Path(f"data/processed/gnn/outage_model_compare{RTAG}{DSFX}.json").write_text(json.dumps(out, indent=1),
                                                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
