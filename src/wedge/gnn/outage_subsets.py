"""Outage responses on subsets of the BritNed and Nemo Link events.

Subsets of the frozen outage_v2 design (same blocks, windows, residuals and doses from the block files; no refit):
  all         every block (reproduces the frozen point estimates; the check is asserted)
  confirmed   blocks of events whose zero flow a published outage message confirms (outage_events_v4.json)
  matched     blocks that received at least one normal-operation window
Estimator: outage_v2.estimate on the subset (pseudo windows restricted to the windows of the subset's blocks), with
1,000 new bootstrap draws over the subset's events. Model side: graph and node-local network-response models and the
two-zone baseline, as in outage_model_compare (effect rho_i(t) L_t in treated hours, zero in windows), on the subset's
hours with identical masks, and the graph model with the network state of the day before each event (encoder window
ending at the same hour of day before the event starts, instead of the outage state). GB side pooled over both links
as in outage_pooled_gb (stratified bootstrap).
Output: data/processed/gnn/outage_subsets{DSFX}.json
Usage: WEDGE_GNN_SPEC=v4 NETRESP_TAG=r2 python src/wedge/gnn/outage_subsets.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import DSFX, Features                                              # noqa: E402
from wedge.gnn.netresp import ETA, Data, intervene, load_models, local_emission_slopes      # noqa: E402
from wedge.gnn.outage_model_compare import model_slopes                                   # noqa: E402
from wedge.gnn.outage_v2 import NB, block_stats, design, drivers, estimate, outcome_ok, slopes   # noqa: E402

DEV = "cuda"
KEYS = ("R_m", "NIother_m", "R_x", "NIother_x", "E_x")
FES = ("A_block", "B_block_hour")


def subset(D, keep_block, rng):
    """Restrict a design to the blocks in keep_block (bool per block), with new bootstrap draws over its events."""
    S = dict(D)
    kb = np.asarray(keep_block, bool)
    S["so_all"] = D["so_all"] & (D["blk"] >= 0) & kb[np.maximum(D["blk"], 0)]
    par = np.where(D["pid"] >= 0, D["par_of_pblock"][np.maximum(D["pid"], 0)], -1)
    S["sp_all"] = D["sp_all"] & (par >= 0) & kb[np.maximum(par, 0)]
    eob, pob = D["ev_of_block"], D["par_of_pblock"]
    ev_ids = np.unique(eob[kb])
    draws = []
    for _ in range(NB):
        es = rng.choice(ev_ids, len(ev_ids))
        bo = np.concatenate([np.where((eob == e) & kb)[0] for e in es])
        bp_ = np.concatenate([np.where(np.isin(pob, np.where((eob == e) & kb)[0]))[0] for e in es])
        draws.append((bo, bp_))
    S["ev_ids"], S["draws"] = ev_ids, draws
    return S


def main():
    fe = Features()
    dat = Data(fe)
    d = np.load(f"data/processed/gnn/dataset{DSFX}.npz")
    idx = {n: i for i, n in enumerate(fe.nodes)}
    _, _, _, _, E, R, NI = drivers(fe, d)
    DS = design(fe, d, np.random.default_rng(11))
    frozen = json.loads(pathlib.Path(f"data/processed/gnn/outage_v2{DSFX}.json").read_text(encoding="utf-8"))
    evl = json.loads(pathlib.Path(f"data/processed/gnn/outage_events{DSFX}.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(111)
    out, pooled_parts = {}, {}
    for b, D in DS.items():
        x, m, xi, mi = D["x"], D["m"], D["xi"], D["mi"]
        F = np.nan_to_num(D["F"])
        z = np.load(f"data/processed/gnn/outage_v2_blocks_{b.replace('->', '_')}{DSFX}.npz")
        so, sp = D["so_all"], D["sp_all"]
        Luse, Fh = np.full(fe.H, np.nan), np.full(fe.H, np.nan)
        Luse[so], Fh[sp] = z["L"], z["p_L"]
        Ys = {"R_m": R[mi], "NIother_m": NI[mi] - F, "R_x": R[xi], "NIother_x": NI[xi] + F, "E_x": E[xi]}
        zkey = {"E_x": f"E_{x}"}
        dYs = {}
        for k in KEYS:
            v = np.zeros(fe.H)
            v[so], v[sp] = z[f"dY_{zkey.get(k, k)}"], z[f"pdY_{zkey.get(k, k)}"]
            dYs[k] = v
        oks = {k: outcome_ok(fe, idx, zkey.get(k, k), D, Ys[k]) for k in KEYS}
        # model effects per treated hour (as outage_model_compare): rho * L, emissions via local responses
        hours, L = z["hours"], z["L"]
        Ls = np.where(np.abs(L) > 1e-6, L, 1e-6)
        msl = local_emission_slopes(fe, hours, range(fe.N), DEV)
        # pre-event state: the parameters of the same hour of day on the day before the event starts
        starts = np.array([D["ev"][e][0] for e in z["event"]])
        hours_pre = starts - 24 + (hours - starts) % 24
        eff = {}
        for name, local, hs in (("graph", False, hours), ("local", True, hours), ("graph_pre_event", False, hours_pre)):
            rho, ni = intervene(load_models(fe, DEV, local), dat, hs, xi, mi, ETA[b], Ls, DEV, return_ni=True)
            dR, dNI = rho * L[:, None], ni * L[:, None]
            eff[name] = {"R_m": dR[:, mi], "NIother_m": dNI[:, mi], "R_x": dR[:, xi], "NIother_x": dNI[:, xi],
                         "E_x": dR[:, xi] * msl[:, xi]}
        pos = L > 0
        dRm, dRx = np.where(pos, L, L / ETA[b]), np.where(pos, -L / ETA[b], -L)
        eff["two_zone"] = {"R_m": dRm, "NIother_m": np.zeros_like(L), "R_x": dRx, "NIother_x": np.zeros_like(L),
                           "E_x": dRx * msl[:, xi]}
        # subsets
        rows = [e for e in evl["events"] if e["link"] == f"{x}-{m}"]
        conf = np.array([rows[e]["confirmed"] for e in D["ev_of_block"]])
        matched = np.isin(np.arange(len(D["blocks"])), np.unique(D["par_of_pblock"]))
        res_b = {}
        for sname, kb in (("all", np.ones(len(D["blocks"]), bool)), ("confirmed", conf), ("matched", matched),
                          ("confirmed_matched", conf & matched)):
            S = subset(D, kb, rng)
            info = {"blocks": int(kb.sum()), "events": int(len(S["ev_ids"])), "treated_hours": int(S["so_all"].sum()),
                    "windows": int(len(np.unique(D["pid"][S["sp_all"]]))), "outcomes": {}}
            hsel = S["so_all"][hours]
            for k in KEYS:
                o_k, bsd = estimate(dYs[k], oks[k], S, Luse, Fh)
                okk = oks[k][hours] & hsel
                mod = {nm: {fes: model_slopes(hours[okk], z["block"][okk], L[okk], ef[k][okk], fes).tolist()
                            for fes in FES} for nm, ef in eff.items()}
                info["outcomes"][k] = {"empirical": {fes: {kk: o_k[fes][kk] for kk in ("outage", "pseudo", "did", "ci95")}
                                                     for fes in FES}, "model": mod}
                if sname == "all":                                     # frozen point estimates reproduced
                    fk = frozen[b]["outcomes"][f"{zkey.get(k, k)}|full"]
                    for fes in FES:
                        assert np.allclose(o_k[fes]["did"], fk[fes]["did"], equal_nan=True), (b, k, fes)
                if k in ("R_x", "NIother_x", "E_x"):
                    pooled_parts.setdefault(sname, {}).setdefault(k, []).append((b, S, dYs[k], oks[k], Luse, Fh,
                                                                                 hours, z["block"], L, okk, eff, k))
            res_b[sname] = info
            print(b, sname, info["events"], "events", info["treated_hours"], "h |",
                  {k: [round(v, 2) for v in info["outcomes"][k]["empirical"]["A_block"]["did"]] for k in KEYS}, flush=True)
        out[b] = res_b
    # GB side pooled over both links (per-block statistics summed, draws stratified by link)
    pooled = {}
    for sname, parts in pooled_parts.items():
        pooled[sname] = {}
        for k, lst in parts.items():
            pooled[sname][k] = {}
            for fes in FES:
                So_all, Sp_all, draws_l, mods = [], [], [], {}
                for (b, S, dY, okY, Luse, Fh, hours, blk, L, okk, eff, kk) in lst:
                    o, p = S["so_all"] & okY, S["sp_all"] & okY
                    go = S["gA_o"] if fes == "A_block" else S["gB_o"]
                    gp = S["gA_p"] if fes == "A_block" else S["gB_p"]
                    _, go_i = np.unique(go[o], return_inverse=True)
                    _, gp_i = np.unique(gp[p], return_inverse=True)
                    So_all.append(block_stats(Luse[o], dY[o], S["blk"][o], go_i, len(S["blocks"])))
                    Sp_all.append(block_stats(Fh[p], dY[p], S["pid"][p], gp_i, len(S["pblocks"])))
                    draws_l.append(S["draws"])
                    for nm, ef in eff.items():
                        g = blk if fes == "A_block" else blk * 24 + hours % 24
                        _, gi = np.unique(g[okk], return_inverse=True)
                        _, bi = np.unique(blk[okk], return_inverse=True)
                        mods.setdefault(nm, []).append(block_stats(L[okk], ef[kk][okk], bi, gi, bi.max() + 1).sum(0))
                b_out = slopes(So_all[0].sum(0) + So_all[1].sum(0))
                b_ps = slopes(Sp_all[0].sum(0) + Sp_all[1].sum(0))
                est = b_out - b_ps
                bs = np.array([slopes(So_all[0][d0[0]].sum(0) + So_all[1][d1[0]].sum(0))
                               - slopes(Sp_all[0][d0[1]].sum(0) + Sp_all[1][d1[1]].sum(0))
                               for d0, d1 in zip(draws_l[0], draws_l[1])])
                pooled[sname][k][fes] = {"outage": b_out.tolist(), "pseudo": b_ps.tolist(), "did": est.tolist(),
                                         "ci95": np.nanpercentile(bs, [2.5, 97.5], axis=0).T.tolist(),
                                         "model": {nm: slopes(v[0] + v[1]).tolist() for nm, v in mods.items()}}
            print("GB pooled", sname, k, [round(v, 2) for v in pooled[sname][k]["A_block"]["did"]],
                  {nm: [round(v, 2) for v in vv] for nm, vv in pooled[sname][k]["A_block"]["model"].items()}, flush=True)
    out["GB_pooled"] = pooled
    pathlib.Path(f"data/processed/gnn/outage_subsets{DSFX}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
