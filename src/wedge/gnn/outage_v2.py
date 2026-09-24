"""Outage natural experiment, revised design (A54).

Changes relative to outage_study.py / outage_did.py
  * treatment mask = hours with zero flow inside a merged event envelope; dose L_t = Fhat_t - F_obs_t;
    hours in which the other focal GB link is also out are dropped; open spells at the sample end are kept;
    spells straddling commissioning keep their post-commissioning part                         (findings 11, 12)
  * events are split into blocks of at most 168 h (clustered by physical event for inference); every block gets
    pseudo blocks of EXACTLY its length, starting in the same calendar quarter and within +-1 year where possible
    (support failures recorded)                                                                 (findings 2, 4, 7)
  * direction-specific slopes: L+ = max(L, 0) (the link would have carried GB exports) and L- = min(L, 0)
    (it would have carried imports into GB), estimated jointly within blocks; pooled slope as well
  * specification B demeans within block x hour-of-day, so identification comes from day-to-day variation of the
    would-be flow only
  * joint bootstrap (1000 draws) over physical events and pseudo blocks, identical draws for every outcome, so balance
    sums and zone contrasts get intervals; per-event weights and leave-one-event-out estimates   (findings 13, 14)
  * REMIT classification of events (planned / unplanned / unclassified) and first-publication lead time
  * reduced-driver sensitivity: load, wind and solar of all zones + fuel prices + calendar only
Estimand (stated in the paper): the within-block response slope of each outcome to the would-be flow of the link, net of
the same slope in matched normal-operation windows; a response to CAPACITY REMOVAL, not to a charge.
Outputs: data/processed/gnn/outage_v2{DSFX}.json and outage_v2_blocks{DSFX}.npz (hours, block, event, L, dY for the
model-side comparison with the identical estimator).
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import DSFX, Features, hour_of                    # noqa: E402
from wedge.gnn.outage_study import COMMISSIONED, LINKS                    # noqa: E402

FOCAL = {"GB->NL": ("GB", "NL"), "GB->BE": ("GB", "BE")}
BLOCK = 168
NB = 1000
T0 = datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp()


def spells_open(zero, min_len=24):
    """Zero-flow spells >= min_len hours, including a spell still open at the end of the sample."""
    out, s = [], None
    for t, v in enumerate(zero):
        if v and s is None:
            s = t
        if not v and s is not None:
            if t - s >= min_len:
                out.append((s, t))
            s = None
    if s is not None and len(zero) - s >= min_len:
        out.append((s, len(zero)))
    return out


def merge(ev, gap=72):
    out = []
    for s, e in ev:
        if out and s - out[-1][1] < gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def oof(X, y, ok, week, dev="cpu"):
    import xgboost as xgb
    pred = np.full(len(y), np.nan)
    folds = week % 5
    for k in range(5):
        tr, te = ok & (folds != k), folds == k
        m = xgb.XGBRegressor(n_estimators=400, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", device=dev, random_state=k, n_jobs=16)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def year_q(h):
    dt = datetime.fromtimestamp(T0 + 3600 * int(h), tz=timezone.utc)
    return dt.year, (dt.month - 1) // 3


def remit_classes():
    """Physical-outage messages of BritNed and Nemo from Elexon REMIT: list of (start_h, end_h, type, publish_h)."""
    f = pathlib.Path("data/raw/remit/remit_ic.jsonl")
    out = {"GB->NL": [], "GB->BE": []}
    if not f.exists():
        return out
    last = {}
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        pid = r.get("participantId")
        if pid not in ("BRITNED", "NEMO1") or (r.get("unavailableCapacity") or 0) < 500:
            continue
        k = r.get("mrid")
        if k not in last or r.get("revisionNumber", 0) >= last[k].get("revisionNumber", 0):
            last[k] = r
    for r in last.values():
        if r.get("eventStatus") == "Dismissed":
            continue
        try:
            s = hour_of(r["eventStartTime"][:19])
            e = hour_of(r["eventEndTime"][:19]) if r.get("eventEndTime") else s + 1
            p = hour_of(r["publishTime"][:19])
        except Exception:                                                     # noqa: BLE001
            continue
        b = "GB->NL" if r["participantId"] == "BRITNED" else "GB->BE"
        out[b].append((s, e, r.get("unavailabilityType"), p))
    return out


def demean(v, g):
    """Subtract group means (g integer labels >= 0)."""
    s = np.bincount(g, weights=v)
    c = np.bincount(g)
    return v - (s / np.maximum(c, 1))[g]


def block_stats(L, y, grp_block, grp_fe, nb):
    """Per-block sufficient statistics of the within-(fe group) regression of y on [L+, L-] and on L (pooled)."""
    lp, lm = np.maximum(L, 0), np.minimum(L, 0)
    xp, xm, xl, yy = demean(lp, grp_fe), demean(lm, grp_fe), demean(L, grp_fe), demean(y, grp_fe)
    S = np.zeros((nb, 7))
    for j, v in enumerate((xp * xp, xp * xm, xm * xm, xp * yy, xm * yy, xl * xl, xl * yy)):
        S[:, j] = np.bincount(grp_block, weights=v, minlength=nb)
    return S


def slopes(S):
    """From summed sufficient statistics -> (beta_plus, beta_minus, beta_pooled)."""
    a, b, c, d, e, f, g = S
    det = a * c - b * b
    bp = (c * d - b * e) / det if det > 0 else np.nan
    bm = (a * e - b * d) / det if det > 0 else np.nan
    bl = g / f if f > 0 else np.nan
    return np.array([bp, bm, bl])


def drivers(fe, d):
    """Driver matrices (full and reduced), week index, loads and outcome series in MW."""
    ci = {c: i for i, c in enumerate(fe.meta["classes"])}
    gen, load = np.nan_to_num(d["gen"]), np.nan_to_num(d["load"])
    t = np.arange(fe.H)
    cal = [np.sin(2 * np.pi * (t % 24) / 24), np.cos(2 * np.pi * (t % 24) / 24), (t // 24) % 7,
           np.sin(2 * np.pi * t / 8766), np.cos(2 * np.pi * t / 8766), t / 8766.0]
    X_full = np.column_stack([a.T for a in [load, gen[:, ci["wind"]], gen[:, ci["solar"]], fe.z_nd]]
                             + [np.nan_to_num(d["glob"])] + cal)
    X_red = np.column_stack([a.T for a in [load, gen[:, ci["wind"]], gen[:, ci["solar"]]]]
                            + [np.nan_to_num(d["glob"])] + cal)
    week = t // 168
    E = (fe.gen_disp * fe.ef_disp[None, :, None]).sum(1) * 1000
    R, NI = fe.resid * 1000, fe.netimp * 1000
    return X_full, X_red, week, load, E, R, NI


def design(fe, d, rng):
    """Events, treated blocks, pseudo windows, nuisance training hours and bootstrap draws of both focal links.
    Links are processed in the order of FOCAL, pseudo windows before draws, so the random stream is that of the
    frozen run."""
    load = np.nan_to_num(d["load"])
    fok = d["flow_ok"] if "flow_ok" in d.files else np.isfinite(fe.F)
    H, idx = fe.H, {n: i for i, n in enumerate(fe.nodes)}
    hod = np.arange(H) % 24
    # zero-flow masks of both focal links (for overlap exclusion)
    zeros, envs = {}, {}
    for b, (x, m) in FOCAL.items():
        F = fe.F[idx[x], idx[m]]
        ok = fok[idx[x], idx[m]] & np.isfinite(F)
        start = hour_of(COMMISSIONED[b]) if b in COMMISSIONED else 0
        z = ok & (np.abs(F) < 1.0)
        z[:start] = False                                    # straddling spells keep their post-commissioning part
        zeros[b] = z
        envs[b] = merge(spells_open(z))
    out = {}
    for b, (x, m) in FOCAL.items():
        xi, mi = idx[x], idx[m]
        F = fe.F[xi, mi]
        other = [o for o in FOCAL if o != b][0]
        ev = envs[b]
        excl = np.zeros(H, bool)
        for s, e in ev:
            excl[max(s - 48, 0):min(e + 24, H)] = True
        start = hour_of(COMMISSIONED[b]) if b in COMMISSIONED else 0
        excl[:start] = True
        normal = np.isfinite(F) & ~zeros[b] & ~excl & (load[xi] > 0) & (load[mi] > 0) & ~zeros[other]
        # treated hours: zero flow inside the envelope, other focal link available
        evid = np.full(H, -1)
        for n_, (s, e) in enumerate(ev):
            evid[s:e] = n_
        treat = (evid >= 0) & zeros[b] & ~zeros[other]
        # blocks of <= 168 h of treated hours within each event
        blk = np.full(H, -1)
        blocks = []                                          # (block id, event id, start, length)
        for n_ in range(len(ev)):
            hs = np.where(treat & (evid == n_))[0]
            for i0 in range(0, len(hs), BLOCK):
                seg = hs[i0:i0 + BLOCK]
                if len(seg) >= 24:
                    blk[seg] = len(blocks)
                    blocks.append((len(blocks), n_, int(seg[0]), int(len(seg))))
        # pseudo blocks: exact length, same quarter, +-1 year where possible
        pid = np.full(H, -1)
        pblocks, relaxed, failed = [], 0, 0
        taken = np.zeros(H, bool)
        norm_cum = np.concatenate([[0], np.cumsum(~normal)])
        starts_all = np.arange(start + 48, H - 1)
        yq = np.array([year_q(h) for h in range(0, H, 24)])                       # daily resolution
        for (bid, eid, s0, ln) in blocks:
            y0, q0 = year_q(s0)
            cand = starts_all[(starts_all + ln < H)]
            okw = (norm_cum[cand + ln] - norm_cum[cand]) == 0
            cand = cand[okw]
            qy = yq[np.minimum(cand // 24, len(yq) - 1)]
            c1 = cand[(qy[:, 1] == q0) & (np.abs(qy[:, 0] - y0) <= 1)]
            pool = c1 if len(c1) >= 20 else cand[qy[:, 1] == q0]
            if len(c1) < 20:
                relaxed += 1
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
        so_all = treat & (blk >= 0)
        sp_all = pid >= 0
        ev_of_block = np.array([e for (_, e, _, _) in blocks])
        par_of_pblock = np.array([p for (_, p, _, _) in pblocks])
        # bootstrap draws (shared by all outcomes): resample physical events, then their pseudo blocks
        draws = []
        ev_ids = np.unique(ev_of_block)
        for _ in range(NB):
            es = rng.choice(ev_ids, len(ev_ids))
            bo = np.concatenate([np.where(ev_of_block == e)[0] for e in es])
            bp_ = np.concatenate([np.where(np.isin(par_of_pblock, np.where(ev_of_block == e)[0]))[0] for e in es])
            draws.append((bo, bp_))
        out[b] = {"x": x, "m": m, "xi": xi, "mi": mi, "F": F, "ev": ev, "evid": evid, "treat": treat, "blk": blk,
                  "blocks": blocks, "pid": pid, "pblocks": pblocks, "relaxed": relaxed, "failed": failed,
                  "train": normal & (pid < 0),               # nuisance training hours: normal, outside pseudo windows
                  "so_all": so_all, "sp_all": sp_all,
                  "gA_o": blk, "gA_p": pid,                  # group labels of the two FE specifications
                  "gB_o": np.where(so_all, blk * 24 + hod, -1), "gB_p": np.where(sp_all, pid * 24 + hod, -1),
                  "ev_of_block": ev_of_block, "par_of_pblock": par_of_pblock, "ev_ids": ev_ids, "draws": draws}
    return out


def estimate(dY, okY, D, Luse, Fh):
    """Outage, pseudo and difference slopes of one outcome, specifications A (block) and B (block x hour).
    Treated hours: within-group regression of the residual dY on the dose Luse = Fhat - F (= Fhat, since F = 0).
    Pseudo windows: the same regression on the would-be flow Fh = Fhat (no dose is removed there).
    Returns the result dict and, per specification, the bootstrap draws of the difference."""
    blk, pid, blocks, pblocks = D["blk"], D["pid"], D["blocks"], D["pblocks"]
    ev_of_block, par_of_pblock, ev_ids, draws = D["ev_of_block"], D["par_of_pblock"], D["ev_ids"], D["draws"]
    o = D["so_all"] & okY
    p = D["sp_all"] & okY
    out_k, bsd = {}, {}
    for fes, go, gp in (("A_block", D["gA_o"], D["gA_p"]), ("B_block_hour", D["gB_o"], D["gB_p"])):
        # relabel groups to 0..n-1 inside the used rows
        _, go_i = np.unique(go[o], return_inverse=True)
        _, gp_i = np.unique(gp[p], return_inverse=True)
        So = block_stats(Luse[o], dY[o], blk[o], go_i, len(blocks))    # (n_blocks, 7)
        Sp = block_stats(Fh[p], dY[p], pid[p], gp_i, len(pblocks))     # (n_pblocks, 7)
        bo_ = slopes(So.sum(0))
        bp_ = slopes(Sp.sum(0))
        did = bo_ - bp_
        bs = np.array([slopes(So[dr[0]].sum(0)) - slopes(Sp[dr[1]].sum(0)) for dr in draws])
        loo = np.array([slopes(So[ev_of_block != e].sum(0)) - slopes(Sp[~np.isin(par_of_pblock,
                        np.where(ev_of_block == e)[0])].sum(0)) for e in ev_ids])
        out_k[fes] = {"outage": bo_.tolist(), "pseudo": bp_.tolist(), "did": did.tolist(),
                      "ci95": np.nanpercentile(bs, [2.5, 97.5], axis=0).T.tolist(),
                      "loo_min": np.nanmin(loo, 0).tolist(), "loo_max": np.nanmax(loo, 0).tolist()}
        bsd[fes] = bs
    out_k["order"] = ["plus (would-be GB export)", "minus (would-be GB import)", "pooled"]
    return out_k, bsd


def outcome_ok(fe, idx, k, D, Y):
    """Hours in which outcome k is observed (both endpoint zones, and the zone of a zonal outcome)."""
    okY = fe.disp_obs_ok[D["mi"]] & fe.disp_obs_ok[D["xi"]] & np.isfinite(Y)
    if k[:2] in ("E_", "R_") and k not in ("R_m", "R_x"):
        okY &= fe.disp_obs_ok[idx[k.split("_", 1)[1]]]
    return okY


def main():
    fe = Features()
    d = np.load(f"data/processed/gnn/dataset{DSFX}.npz")
    idx = {n: i for i, n in enumerate(fe.nodes)}
    X_full, X_red, week, load, E, R, NI = drivers(fe, d)
    remit = remit_classes()
    res, rng = {}, np.random.default_rng(11)
    DS = design(fe, d, rng)
    for b, D in DS.items():
        x, m, xi, mi, F = D["x"], D["m"], D["xi"], D["mi"], D["F"]
        ev, evid, treat, blk, blocks = D["ev"], D["evid"], D["treat"], D["blk"], D["blocks"]
        pid, pblocks, train, so_all, sp_all = D["pid"], D["pblocks"], D["train"], D["so_all"], D["sp_all"]
        n_ev = len(ev)
        # nuisance predictions (trained on normal hours outside pseudo windows)
        Fhat = oof(X_full, np.nan_to_num(F), train, week)
        Fhat_red = oof(X_red, np.nan_to_num(F), train, week)
        L = Fhat - np.where(treat, np.nan_to_num(F), 0.0)                    # dose; F ~ 0 in treated hours
        Lp = Fhat.copy()                                                     # pseudo windows: would-be flow (no dose)
        outcomes = {"R_m": R[mi], "R_x": R[xi], "NIother_m": NI[mi] - np.nan_to_num(F),
                    "NIother_x": NI[xi] + np.nan_to_num(F)}
        for n in fe.nodes:
            outcomes[f"E_{n}"] = E[idx[n]]
            outcomes[f"R_{n}"] = R[idx[n]]
        key_red = {"R_m", "R_x", "NIother_m", "NIother_x", f"E_{x}", f"E_{m}", "E_DE"}
        keep = {"hours": np.where(so_all)[0], "block": blk[so_all], "event": evid[so_all], "L": L[so_all],
                "p_hours": np.where(sp_all)[0], "p_block": pid[sp_all], "p_parent": D["par_of_pblock"][pid[sp_all]],
                "p_L": Lp[sp_all]}
        r = {"events": n_ev, "blocks": len(blocks), "pseudo_blocks": len(pblocks), "pseudo_relaxed": D["relaxed"],
             "pseudo_failed": D["failed"], "treated_hours": int(so_all.sum()),
             "share_hours_L_positive": float((L[so_all] > 0).mean()),
             "events_table": [], "outcomes": {}}
        # event table with REMIT classification and identifying weight
        w_ev = np.zeros(n_ev)
        for (bid, eid, s0, ln) in blocks:
            hs = np.where(blk == bid)[0]
            w_ev[eid] += ((L[hs] - L[hs].mean()) ** 2).sum()
        w_ev = w_ev / w_ev.sum()
        for n_, (s, e) in enumerate(ev):
            typ, lead = "unclassified", None
            ov = [(ms, me, ty, pu) for (ms, me, ty, pu) in remit[b] if ms < e and me > s]
            if ov:
                hrs = {}
                for ms, me, ty, pu in ov:
                    hrs[ty] = hrs.get(ty, 0) + min(me, e) - max(ms, s)
                typ = max(hrs, key=hrs.get)
                lead = float((s - min(pu for (_, _, _, pu) in ov)) / 24.0)
            dt = datetime.fromtimestamp(T0 + 3600 * s, tz=timezone.utc).strftime("%Y-%m-%d")
            r["events_table"].append({"event": n_, "start": dt, "hours": int(e - s),
                                      "treated_hours": int((so_all & (evid == n_)).sum()),
                                      "weight": float(w_ev[n_]), "remit": typ, "lead_days": lead,
                                      "mean_L": float(np.nanmean(L[so_all & (evid == n_)])) if (so_all & (evid == n_)).any() else None})
        r["effective_events"] = float(1.0 / (w_ev ** 2).sum())
        allB = {}
        for k, Y in outcomes.items():
            okY = outcome_ok(fe, idx, k, D, Y)
            for spec, X_, Fh in (("full", X_full, Fhat), ("reduced", X_red, Fhat_red)):
                if spec == "reduced" and k not in key_red:
                    continue
                Yhat = oof(X_, np.nan_to_num(Y), okY & train, week)
                dY = np.nan_to_num(Y - Yhat)
                Luse = Fh - np.where(treat, np.nan_to_num(F), 0.0)
                out_k, bsd = estimate(dY, okY, D, Luse, Fh)
                if spec == "full":
                    for fes, bs in bsd.items():
                        allB[(k, fes)] = bs
                r["outcomes"][f"{k}|{spec}"] = out_k
                if spec == "full":
                    keep[f"dY_{k}"] = dY[so_all]
                    keep[f"pdY_{k}"] = dY[sp_all]
                print(b, k, spec, "A did +/-/pool", np.round(out_k["A_block"]["did"], 2),
                      "B", np.round(out_k["B_block_hour"]["did"], 2), flush=True)
        # joint quantities from the shared draws (spec full)
        joint = {}
        for fes in ("A_block", "B_block_hour"):
            bal_m = allB[("R_m", fes)] + allB[("NIother_m", fes)]
            bal_x = allB[("R_x", fes)] + allB[("NIother_x", fes)]
            e_keys = [kk for (kk, f_) in allB if f_ == fes and kk.startswith("E_")]
            e_in = allB[(f"E_{m}", fes)]
            e_out = sum(allB[(kk, fes)] for kk in e_keys if kk not in (f"E_{m}", f"E_{x}"))
            e_net = sum(allB[(kk, fes)] for kk in e_keys)
            joint[fes] = {"balance_importer": np.nanpercentile(bal_m, [2.5, 50, 97.5], axis=0).T.tolist(),
                          "balance_exporter": np.nanpercentile(bal_x, [2.5, 50, 97.5], axis=0).T.tolist(),
                          "emis_outside_minus_importer": np.nanpercentile(e_out - e_in, [2.5, 50, 97.5], axis=0).T.tolist(),
                          "emis_network": np.nanpercentile(e_net, [2.5, 50, 97.5], axis=0).T.tolist()}
        r["joint"] = joint
        res[b] = r
        np.savez_compressed(f"data/processed/gnn/outage_v2_blocks_{b.replace('->', '_')}{DSFX}.npz", **keep)
    pathlib.Path(f"data/processed/gnn/outage_v2{DSFX}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
