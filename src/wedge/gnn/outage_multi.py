"""Held-out outage validation of the network-response model on further HVDC interconnectors (A59).

The network-response model is trained on 2019-2022. Outages of HVDC links that start in 2023 or later were therefore
never seen by it, and each of them removes one link while the rest of the network stays available, as the BritNed
and Nemo outages do. For every HVDC link of the graph with at least three such outages, this script applies the
estimator of outage_v2.py and the model comparison of outage_model_compare.py:
  events      spells of at least 24 h with zero flow in both directions (flow_ok in both), merged across gaps < 72 h,
              starting on or after 2023-01-01; treated hours exclude hours in which another candidate link that
              shares an endpoint is also out
  dose        L = nhat - n, n = F[a,b] - F[b,a] the net flow a->b, nhat its out-of-fold prediction from exogenous
              drivers (outage_v2 X_full); L+ = would-be flow a->b, L- = would-be flow b->a
  outcomes    R_a, R_b (dispatchable output), NIother_a = NI_a + n, NIother_b = NI_b - n (net imports on the other
              links), E_a, E_b (dispatchable emissions)
  estimator   within-block slopes on [L+, L-] and on L, blocks <= 168 h, five exact-length pseudo windows per block
              from the same quarter within one year, difference in slopes, specifications A (block) and B
              (block x hour), 1,000 bootstrap draws over events (identical draws for all outcomes)
  model side  netresp intervene with the same L on the same hours (graph and node-local encoders, three models each),
              other-link net imports from the solved internal flows, emissions via the dispatch head's local
              responses; two-zone accounting dR_b = L, dR_a = -L/eta for L > 0 (sender losses) and symmetrically
              for L < 0; identical outcome masks and estimator
Loss factor eta = 0.97 for every HVDC link (it scales the two-zone exporter response only).
Output: data/processed/gnn/outage_multi_v4.json
Usage: WEDGE_GNN_SPEC=v4 NETRESP_TAG=r2 python src/wedge/gnn/outage_multi.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import DSFX, Features, hour_of                                  # noqa: E402
from wedge.gnn.outage_v2 import BLOCK, NB, block_stats, merge, oof, slopes, spells_open, year_q   # noqa: E402

HVDC = [("DE", "SE"), ("GR", "IT"), ("NL", "NO"), ("PL", "SE"), ("DK", "NL"), ("ME", "IT"), ("BE", "DE"),
        ("DE", "NO"), ("GB", "NO"), ("DK", "GB")]
COMMISSIONED = {("DK", "GB"): "2023-12-29"}          # Viking Link commercial operation
# A60: events start after the January-June 2023 half-year on which both model
# components selected their checkpoints, so the test outages are untouched by training and selection
HELD = "2023-07-01"
VERSION = "A60-fix1"                                 # F is antisymmetric (F[b,a] = -F[a,b]); n = F[a,b]
ETA = 0.97
DEV = "cuda"
MIN_EVENTS = 3


def main(links=None, held_start=None, out_name="outage_multi", event_ok=None, eta=None, min_events=None):
    """links: list of (a, b); held_start: first allowed event start (ISO date); out_name: artefact stem;
    event_ok: optional function (a, b, start_hour, end_hour) -> bool that keeps only verified events (A62);
    eta: loss factor for the two-zone comparator; min_events: minimum number of events per link.
    The defaults reproduce the A59/A60-fix1 run over the HVDC links."""
    links = HVDC if links is None else links
    held_start = HELD if held_start is None else held_start
    eta = ETA if eta is None else eta
    min_events = MIN_EVENTS if min_events is None else min_events
    fe = Features()
    d = np.load(f"data/processed/gnn/dataset{DSFX}.npz")
    ci = {c: i for i, c in enumerate(fe.meta["classes"])}
    gen, load = np.nan_to_num(d["gen"]), np.nan_to_num(d["load"])
    fok = d["flow_ok"] if "flow_ok" in d.files else np.isfinite(fe.F)
    H, idx, t = fe.H, {n: i for i, n in enumerate(fe.nodes)}, np.arange(fe.H)
    cal = [np.sin(2 * np.pi * (t % 24) / 24), np.cos(2 * np.pi * (t % 24) / 24), (t // 24) % 7,
           np.sin(2 * np.pi * t / 8766), np.cos(2 * np.pi * t / 8766), t / 8766.0]
    X_full = np.column_stack([a.T for a in [load, gen[:, ci["wind"]], gen[:, ci["solar"]], fe.z_nd]]
                             + [np.nan_to_num(d["glob"])] + cal)
    week = t // 168
    E = (fe.gen_disp * fe.ef_disp[None, :, None]).sum(1) * 1000
    R, NI = fe.resid * 1000, fe.netimp * 1000
    hod = t % 24
    held = hour_of(held_start)
    # zero-flow masks of all candidate links (the HVDC links always enter the common-endpoint exclusions)
    zeros, nets, oks = {}, {}, {}
    for a, b in list(dict.fromkeys(list(links) + HVDC)):
        ai, bi = idx[a], idx[b]
        F1 = fe.F[ai, bi]                                 # signed net flow a -> b (the dataset stores F[b,a] = -F[a,b])
        ok = fok[ai, bi] & np.isfinite(F1)
        tot = np.abs(np.nan_to_num(F1))
        first = np.where(ok & (tot > 1.0))[0]
        z = ok & (tot < 1.0)
        if len(first):
            z[:first[0]] = False
        if (a, b) in COMMISSIONED:
            z[:hour_of(COMMISSIONED[(a, b)])] = False
        zeros[(a, b)] = z
        nets[(a, b)] = np.nan_to_num(F1)
        oks[(a, b)] = ok
    rng = np.random.default_rng(29)
    res, keep_all = {}, {}
    for a, b in links:
        ai, bi = idx[a], idx[b]
        ev_all = merge(spells_open(zeros[(a, b)]))
        ev = [(s, e) for s, e in ev_all if s >= held]
        if event_ok is not None:
            ev = [(s, e) for s, e in ev if event_ok(a, b, s, e)]
        if len(ev) < min_events:
            print(a, b, "events", len(ev), "skip", flush=True)
            continue
        others = [k for k in list(dict.fromkeys(list(links) + HVDC)) if k != (a, b) and (a in k or b in k)]
        other_zero = np.zeros(H, bool)
        for k in others:
            other_zero |= zeros[k]
        excl = np.zeros(H, bool)
        for s, e in ev_all:
            excl[max(s - 48, 0):min(e + 24, H)] = True
        n = nets[(a, b)]
        normal = oks[(a, b)] & ~zeros[(a, b)] & ~excl & (load[ai] > 0) & (load[bi] > 0) & ~other_zero
        evid = np.full(H, -1)
        for k_, (s, e) in enumerate(ev):
            evid[s:e] = k_
        treat = (evid >= 0) & zeros[(a, b)] & ~other_zero
        blk = np.full(H, -1)
        blocks = []
        for k_ in range(len(ev)):
            hs = np.where(treat & (evid == k_))[0]
            for i0 in range(0, len(hs), BLOCK):
                seg = hs[i0:i0 + BLOCK]
                if len(seg) >= 24:
                    blk[seg] = len(blocks)
                    blocks.append((len(blocks), k_, int(seg[0]), int(len(seg))))
        if len(blocks) < 2:
            print(a, b, "blocks", len(blocks), "skip", flush=True)
            continue
        pid = np.full(H, -1)
        pblocks, relaxed, failed = [], 0, 0
        taken = np.zeros(H, bool)
        norm_cum = np.concatenate([[0], np.cumsum(~normal)])
        starts_all = np.arange(48, H - 1)
        yq = np.array([year_q(h) for h in range(0, H, 24)])
        for (bid, eid, s0, ln) in blocks:
            y0, q0 = year_q(s0)
            cand = starts_all[(starts_all + ln < H)]
            cand = cand[(norm_cum[cand + ln] - norm_cum[cand]) == 0]
            qy = yq[np.minimum(cand // 24, len(yq) - 1)]
            c1 = cand[(qy[:, 1] == q0) & (np.abs(qy[:, 0] - y0) <= 1)]
            pool = c1 if len(c1) >= 20 else cand[qy[:, 1] == q0]
            relaxed += int(len(c1) < 20)
            got = 0
            for s in rng.permutation(pool):
                if got == 5:
                    break
                if not taken[s:s + ln].any():
                    pid[s:s + ln] = len(pblocks)
                    taken[s:s + ln] = True
                    pblocks.append((len(pblocks), bid, int(s), int(ln)))
                    got += 1
            failed += 5 - got
        train = normal & (pid < 0)
        nhat = oof(X_full, n, train, week, dev=DEV)
        L = np.where(treat, nhat - n, 0.0)
        so_all, sp_all = treat & (blk >= 0), pid >= 0
        ev_of_block = np.array([e for (_, e, _, _) in blocks])
        par_of_pblock = np.array([p for (_, p, _, _) in pblocks])
        ev_ids = np.unique(ev_of_block)
        draws = []
        for _ in range(NB):
            es = rng.choice(ev_ids, len(ev_ids))
            bo = np.concatenate([np.where(ev_of_block == e)[0] for e in es])
            bp_ = np.concatenate([np.where(np.isin(par_of_pblock, np.where(ev_of_block == e)[0]))[0] for e in es])
            draws.append((bo, bp_))
        outcomes = {"R_a": R[ai], "R_b": R[bi], "NIother_a": NI[ai] + n, "NIother_b": NI[bi] - n,
                    "E_a": E[ai], "E_b": E[bi]}
        w_ev = np.zeros(len(ev))
        for (bid, eid, s0, ln) in blocks:
            hs = np.where(blk == bid)[0]
            w_ev[eid] += ((L[hs] - L[hs].mean()) ** 2).sum()
        w_ev = w_ev / w_ev.sum()
        r = {"link": f"{a}-{b}", "events": len(ev), "events_with_blocks": int(len(ev_ids)), "blocks": len(blocks),
             "pseudo_blocks": len(pblocks), "pseudo_relaxed": relaxed, "pseudo_failed": failed,
             "blocks_with_window": int(len(np.unique(par_of_pblock))) if len(pblocks) else 0,
             "treated_hours": int(so_all.sum()), "effective_events": float(1.0 / (w_ev ** 2).sum()),
             "share_hours_L_positive": float((L[so_all] > 0).mean()), "mean_abs_L": float(np.abs(L[so_all]).mean()),
             "flow_slope_normal": None, "outcomes": {}}
        # calibration of the dose at the margin the estimator uses: within-window slope of n on nhat in pseudo windows
        pw = sp_all & oks[(a, b)]
        if pw.sum() > 100:
            from wedge.gnn.outage_v2 import demean
            xx, yy = demean(nhat[pw], pid[pw]), demean(n[pw], pid[pw])
            r["flow_slope_normal"] = float((xx @ yy) / (xx @ xx))
        keep = {"hours": np.where(so_all)[0], "block": blk[so_all], "event": evid[so_all], "L": L[so_all],
                "p_hours": np.where(sp_all)[0], "p_block": pid[sp_all], "p_parent": par_of_pblock[pid[sp_all]],
                "p_L": nhat[sp_all]}                               # A64: would-be flow in the windows (per-event use)
        for k, Y in outcomes.items():
            zi = ai if k.endswith("_a") else bi
            okY = fe.disp_obs_ok[ai] & fe.disp_obs_ok[bi] & np.isfinite(Y)
            Yhat = oof(X_full, np.nan_to_num(Y), okY & train, week, dev=DEV)
            dY = np.nan_to_num(Y - Yhat)
            o, p = so_all & okY, sp_all & okY
            out_k = {"n_hours": int(o.sum())}
            for fes, go, gp in (("A_block", blk, pid), ("B_block_hour", np.where(so_all, blk * 24 + hod, -1),
                                                           np.where(sp_all, pid * 24 + hod, -1))):
                _, go_i = np.unique(go[o], return_inverse=True)
                _, gp_i = np.unique(gp[p], return_inverse=True)
                So = block_stats(L[o], dY[o], blk[o], go_i, len(blocks))
                Sp = block_stats(nhat[p], dY[p], pid[p], gp_i, len(pblocks))
                did = slopes(So.sum(0)) - slopes(Sp.sum(0))
                bs = np.array([slopes(So[dr[0]].sum(0)) - slopes(Sp[dr[1]].sum(0)) for dr in draws])
                out_k[fes] = {"did": did.tolist(), "ci95": np.nanpercentile(bs, [2.5, 97.5], axis=0).T.tolist()}
            r["outcomes"][k] = out_k
            keep[f"dY_{k}"] = dY[so_all]
            keep[f"ok_{k}"] = okY[so_all]
            keep[f"pdY_{k}"] = dY[sp_all]                          # A64: window residuals and masks
            keep[f"pok_{k}"] = okY[sp_all]
            print(a, b, k, "A did", np.round(out_k["A_block"]["did"], 2), "ci", np.round(out_k["A_block"]["ci95"], 2).tolist(),
                  flush=True)
        res[f"{a}-{b}"] = r
        keep_all[f"{a}-{b}"] = keep
        np.savez_compressed(f"data/processed/gnn/{out_name}_blocks_{a}_{b}{DSFX}.npz", **keep)
    pathlib.Path(f"data/processed/gnn/{out_name}{DSFX}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    model_side(fe, res, keep_all, idx, out_name=out_name, eta=eta)


def model_slopes(hours, blk, L, dY, fes):
    hod = hours % 24
    g = blk if fes == "A_block" else blk * 24 + hod
    _, gi = np.unique(g, return_inverse=True)
    _, bi = np.unique(blk, return_inverse=True)
    S = block_stats(L, dY, bi, gi, bi.max() + 1)
    return slopes(S.sum(0))


def model_side(fe, res, keep_all, idx, out_name="outage_multi", eta=ETA):
    from wedge.gnn.netresp import Data, intervene, load_models, local_emission_slopes
    dat = Data(fe)
    models = {"graph": load_models(fe, DEV, False)}
    try:
        models["local"] = load_models(fe, DEV, True)
    except FileNotFoundError:                                                      # A60: r3 node-local may lag
        print("no node-local checkpoints, skipped", flush=True)
    for key, r in res.items():
        a, b = key.split("-")
        ai, bi = idx[a], idx[b]
        z = keep_all[key]
        hours, blk, L = z["hours"], z["block"], z["L"]
        Ls = np.where(np.abs(L) > 1e-6, L, 1e-6)
        ms = local_emission_slopes(fe, hours, [ai, bi], DEV)                       # (T, N), columns ai and bi filled
        resp = {}
        for name, mdl in models.items():
            rho, ni = intervene(mdl, dat, hours, ai, bi, eta, Ls, DEV, return_ni=True)
            resp[name] = (rho * L[:, None], ni * L[:, None])
        pos = L > 0
        dR1 = np.zeros((len(hours), fe.N))
        dR1[:, bi] = np.where(pos, L, L / eta)
        dR1[:, ai] = np.where(pos, -L / eta, -L)
        resp["one_for_one"] = (dR1, np.zeros_like(dR1))
        r["model"] = {}
        effects = {}                                               # A64: per-hour model effects for per-event slopes
        for name, (dR, dNI) in resp.items():
            ys = {"R_a": dR[:, ai], "R_b": dR[:, bi], "NIother_a": dNI[:, ai], "NIother_b": dNI[:, bi],
                  "E_a": dR[:, ai] * ms[:, ai], "E_b": dR[:, bi] * ms[:, bi]}
            for k, y in ys.items():
                effects[f"{name}__{k}"] = np.asarray(y, np.float64)
            rows = {}
            for k, y in ys.items():
                okk = z[f"ok_{k}"].astype(bool)
                rows[k] = {}
                for fes in ("A_block", "B_block_hour"):
                    mv = model_slopes(hours[okk], blk[okk], L[okk], y[okk], fes)
                    ci = np.array(r["outcomes"][k][fes]["ci95"])
                    inside = [bool(ci[j, 0] <= mv[j] <= ci[j, 1]) if np.all(np.isfinite(ci[j])) and np.isfinite(mv[j])
                              else None for j in range(3)]
                    rows[k][fes] = {"model": mv.tolist(), "inside": inside}
            r["model"][name] = rows
        print(key, {name: {k: np.round(r["model"][name][k]["A_block"]["model"], 2).tolist() for k in ("R_b", "NIother_b")}
                    for name in r["model"]}, flush=True)
        import os as _os
        _tag = _os.environ.get("NETRESP_TAG", "r2")
        _rtag = "" if _tag in ("", "r2") else f"_{_tag}"
        np.savez_compressed(f"data/processed/gnn/{out_name}_model_{a}_{b}{_rtag}{DSFX}.npz", hours=hours, **effects)
    # summary over links: pooled-direction responses, specification A
    summ = {}
    for fes in ("A_block", "B_block_hour"):
        for name in [n for n in ("graph", "local", "one_for_one") if n in models or n == "one_for_one"]:
            for grp, keys in (("R", ("R_a", "R_b", "NIother_a", "NIother_b")), ("E", ("E_a", "E_b"))):
                errs, ins = [], []
                for key, r in res.items():
                    if "model" not in r:
                        continue
                    for k in keys:
                        e = r["outcomes"][k][fes]["did"][2]
                        m = r["model"][name][k][fes]["model"][2]
                        if np.isfinite(e) and np.isfinite(m):
                            errs.append(m - e)
                            ins.append(r["model"][name][k][fes]["inside"][2])
                summ[f"{fes}|{name}|{grp}"] = {"rmse": float(np.sqrt(np.mean(np.square(errs)))) if errs else None,
                                               "inside_share": float(np.mean([x for x in ins if x is not None]))
                                               if ins else None, "n": len(errs)}
    res["_summary"] = summ
    res["_version"] = VERSION
    print({k: (round(v["rmse"], 3) if v["rmse"] else None, round(v["inside_share"], 2) if v["inside_share"] else None,
               v["n"]) for k, v in summ.items()}, flush=True)
    import os
    tag = os.environ.get("NETRESP_TAG", "r2")
    rtag = "" if tag in ("", "r2") else f"_{tag}"                                  # A60: r3 must not overwrite r2
    pathlib.Path(f"data/processed/gnn/{out_name}{rtag}{DSFX}.json").write_text(json.dumps(res, indent=1),
                                                                                 encoding="utf-8")


def model_only():
    """Re-run only the model side (e.g. for another network-response tag) on the saved empirical estimates and
    block files of the r2 run."""
    import time
    f = pathlib.Path(f"data/processed/gnn/outage_multi{DSFX}.json")
    for _ in range(180):                                  # wait for the corrected empirical run (A60-fix1)
        if f.exists() and json.loads(f.read_text(encoding="utf-8")).get("_version") == VERSION:
            break
        time.sleep(60)
    else:
        raise RuntimeError("corrected outage_multi run not found")
    fe = Features()
    idx = {n: i for i, n in enumerate(fe.nodes)}
    res = json.loads(f.read_text(encoding="utf-8"))
    res = {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in res.items() if not k.startswith("_")}
    keep_all = {}
    for key in res:
        a, b = key.split("-")
        z = np.load(f"data/processed/gnn/outage_multi_blocks_{a}_{b}{DSFX}.npz")
        keep_all[key] = {k: z[k] for k in z.files}
    model_side(fe, res, keep_all, idx)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "model":
        model_only()
    else:
        main()
