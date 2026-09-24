"""GB-side outage responses pooled over both focal links (A58).

The exporter's response to the removal of one link is a property of the GB system, and the two focal links are of
the same size and connect GB to zones at the same carbon price. Pooling the blocks of both links (outage_v2 block
files, identical outcome masks) tightens the GB-side intervals that each link alone leaves wide.

Estimator: exactly that of outage_v2 (within-block slopes of dY on [L+, L-] and on L, specifications A and B,
difference between outage blocks and matched normal-operation windows), with the per-block sufficient statistics of
both links summed. Bootstrap: 1,000 draws that resample physical events within each link (stratified), each event
with its pseudo blocks; leave-one-event-out over all events of both links.
Model side: in pseudo windows the model effect is zero, and in treated hours it is rho_i(t) L_t, so the pooled model
slope follows exactly from each link's model slopes in outage_model_compare_v4.json and the dose moments of that
link: for the direction-specific slopes (d, e) = [[a, b], [b, c]] (beta_plus, beta_minus), and for the pooled slope
g = f beta, with (a, b, c, f) the within-group moments of the dose on the identical rows.
Outcomes: R_x (GB dispatchable output), NIother_x (GB net imports on its other links), E_GB (GB dispatchable
emissions), and the GB balance R_x + NIother_x.
Output: data/processed/gnn/outage_pooled_gb_v4.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import DSFX, Features                         # noqa: E402
from wedge.gnn.outage_v2 import FOCAL, NB, block_stats, demean, slopes  # noqa: E402
from wedge.gnn.outage_v2_post import outcome_mask                      # noqa: E402

OUT = ("R_x", "NIother_x", "E_GB")


def link_stats(fe, idx, b, k, fes):
    """Per-block sufficient statistics for outcome k on link b (event and pseudo), with block -> event maps."""
    x, m = FOCAL[b]
    z = np.load(f"data/processed/gnn/outage_v2_blocks_{b.replace('->', '_')}{DSFX}.npz")
    hours, blk, ev, L = z["hours"], z["block"], z["event"], z["L"]
    ph, pb, ppar, pL = z["p_hours"], z["p_block"], z["p_parent"], z["p_L"]
    dY, pdY = z[f"dY_{k}"], z[f"pdY_{k}"]
    o = outcome_mask(fe, idx, k, x, m, hours)
    p = outcome_mask(fe, idx, k, x, m, ph)
    nb = int(blk.max()) + 1
    npb = int(pb.max()) + 1
    go = blk if fes == "A_block" else blk * 24 + hours % 24
    gp = pb if fes == "A_block" else pb * 24 + ph % 24
    _, go_i = np.unique(go[o], return_inverse=True)
    _, gp_i = np.unique(gp[p], return_inverse=True)
    So = block_stats(L[o], dY[o], blk[o], go_i, nb)
    Sp = block_stats(pL[p], pdY[p], pb[p], gp_i, npb)
    ev_of_block = np.zeros(nb, int)
    for bb, ee in zip(blk, ev):
        ev_of_block[bb] = ee
    par_of_pblock = np.zeros(npb, int)
    for bb, pp in zip(pb, ppar):
        par_of_pblock[bb] = pp
    # dose moments on the identical event rows (for the model side)
    lp, lm = np.maximum(L[o], 0), np.minimum(L[o], 0)
    xp, xm, xl = demean(lp, go_i), demean(lm, go_i), demean(L[o], go_i)
    mom = np.array([xp @ xp, xp @ xm, xm @ xm, xl @ xl])
    return So, Sp, ev_of_block, par_of_pblock, mom


def main():
    fe = Features()
    idx = {n: i for i, n in enumerate(fe.nodes)}
    import os
    tag = os.environ.get("NETRESP_TAG", "r2")
    rtag = "" if tag in ("", "r2") else f"_{tag}"
    cmp_ = json.loads(pathlib.Path(f"data/processed/gnn/outage_model_compare{rtag}{DSFX}.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(23)
    links = list(FOCAL)
    # stratified event draws, shared by all outcomes and specifications
    base = {b: link_stats(fe, idx, b, "R_x", "A_block") for b in links}
    ev_ids = {b: np.unique(base[b][2]) for b in links}
    draws = [{b: rng.choice(ev_ids[b], len(ev_ids[b])) for b in links} for _ in range(NB)]
    res = {"events": {b: int(len(ev_ids[b])) for b in links}, "order": ["plus", "minus", "pooled"], "rows": {}}
    allB = {}
    for k in OUT:
        res["rows"][k] = {}
        for fes in ("A_block", "B_block_hour"):
            st = {b: link_stats(fe, idx, b, k, fes) for b in links}
            So = sum(st[b][0].sum(0) for b in links)
            Sp = sum(st[b][1].sum(0) for b in links)
            did = slopes(So) - slopes(Sp)

            def pick(b, es):
                So_b, Sp_b, evb, parb, _ = st[b]
                bo = np.concatenate([np.where(evb == e)[0] for e in es])
                bp = np.concatenate([np.where(np.isin(parb, np.where(evb == e)[0]))[0] for e in es])
                return So_b[bo].sum(0), Sp_b[bp].sum(0)

            bs = []
            for dr in draws:
                parts = [pick(b, dr[b]) for b in links]
                bs.append(slopes(sum(p_[0] for p_ in parts)) - slopes(sum(p_[1] for p_ in parts)))
            bs = np.array(bs)
            allB[(k, fes)] = bs
            loo = []
            for b in links:
                for e in ev_ids[b]:
                    parts = [pick(bb, ev_ids[bb][ev_ids[bb] != e] if bb == b else ev_ids[bb]) for bb in links]
                    loo.append(slopes(sum(p_[0] for p_ in parts)) - slopes(sum(p_[1] for p_ in parts)))
            loo = np.array(loo)
            # model side: exact pooled slopes from per-link model slopes and dose moments
            model = {}
            for name in [n for n in ("graph", "local", "one_for_one") if all(n in cmp_[b] for b in links)]:
                d_ = e_ = g_ = 0.0
                a_ = b_ = c_ = f_ = 0.0
                for b in links:
                    bp_, bm_, bl_ = cmp_[b][name]["rows"][k][fes]["model"]
                    a, bb_, c, f = st[b][4]
                    bp_ = 0.0 if not np.isfinite(bp_) else bp_
                    bm_ = 0.0 if not np.isfinite(bm_) else bm_
                    d_ += a * bp_ + bb_ * bm_
                    e_ += bb_ * bp_ + c * bm_
                    g_ += f * bl_
                    a_ += a
                    b_ += bb_
                    c_ += c
                    f_ += f
                det = a_ * c_ - b_ * b_
                model[name] = [(c_ * d_ - b_ * e_) / det, (a_ * e_ - b_ * d_) / det, g_ / f_]
            ci = np.nanpercentile(bs, [2.5, 97.5], axis=0).T
            res["rows"][k][fes] = {"did": did.tolist(), "ci95": ci.tolist(), "loo_min": np.nanmin(loo, 0).tolist(),
                                   "loo_max": np.nanmax(loo, 0).tolist(), "model": model,
                                   "inside": {n: [bool(ci[j, 0] <= v[j] <= ci[j, 1]) for j in range(3)]
                                              for n, v in model.items()}}
            print(k, fes, "did", np.round(did, 2), "ci", np.round(ci, 2).tolist(),
                  {n: np.round(v, 2).tolist() for n, v in model.items()}, flush=True)
    for fes in ("A_block", "B_block_hour"):
        bal = allB[("R_x", fes)] + allB[("NIother_x", fes)]
        pt = np.array(res["rows"]["R_x"][fes]["did"]) + np.array(res["rows"]["NIother_x"][fes]["did"])
        res.setdefault("balance_exporter", {})[fes] = {"did": pt.tolist(),
                                                       "ci95": np.nanpercentile(bal, [2.5, 97.5], axis=0).T.tolist()}
        # share of the lost export that GB plants absorb, R_x / (R_x + NIother_x), from the shared draws
        sh = allB[("R_x", fes)] / (allB[("R_x", fes)] + allB[("NIother_x", fes)])
        res.setdefault("gb_plant_share", {})[fes] = {
            "point": (np.array(res["rows"]["R_x"][fes]["did"]) / pt).tolist(),
            "ci95": np.nanpercentile(sh, [2.5, 97.5], axis=0).T.tolist()}
    pathlib.Path(f"data/processed/gnn/outage_pooled_gb{rtag}{DSFX}.json").write_text(json.dumps(res, indent=1),
                                                                                encoding="utf-8")


if __name__ == "__main__":
    main()
