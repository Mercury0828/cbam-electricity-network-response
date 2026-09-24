"""Physical feasibility of the 100 MW interventions of the network-response model.

For every hour of the charged-border sample of netresp.charged (flow above 100 MW, all zones observed, 2025 and
January to August 2026) and every graph model, the intervention is solved again (g_l = 0 on the charged link, 100 MW
of delivered flow removed, sender losses) and the implied state is compared with physical limits:
  links   perturbed flow |F_e + dF_e| against the capacity proxy of the model (99th percentile of |F_e| over the sample);
          a violation is a change that raises |F_e| above the proxy
  zones   perturbed dispatchable output R_i + dR_i against the range of R_i observed in the same calendar month
          (a violation is a change that leaves that range), and against zero
Also reported: the share of the absolute flow change that falls on links already within 100 MW of the proxy in the
direction of the change, and the share of hours in which the charged link itself carried less than 100 MW more than
its lower limit (none by construction of the sample).
A64 (fourth review): the share of hours in which some link's base flow already lies beyond its proxy, and the
charged-period decomposition of net_summary.py (importer and exporter replacement, E_x, E_m, third zones, network)
recomputed on the hours without a new proxy crossing, and without a new crossing or zone-range exit, per model.
Output: data/processed/gnn/netresp_{tag}_v4/net_feasibility.json
Usage: WEDGE_GNN_SPEC=v4 NETRESP_TAG=r2 python src/wedge/gnn/net_feasibility.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import SPLITS, Features, hour_of                           # noqa: E402
from wedge.gnn.netresp import ETA, OUTD, Data, load_models, solve                  # noqa: E402
from wedge.gnn.outage_study import LINKS                                          # noqa: E402

DEV = "cuda"
DELTA = 100.0
T0 = datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp()


def month_index(H):
    return np.array([(lambda d: d.year * 12 + d.month)(datetime.fromtimestamp(T0 + 3600 * h, tz=timezone.utc))
                     for h in range(H)])


def main():
    fe = Features()
    dat = Data(fe)
    idx = {n: i for i, n in enumerate(fe.nodes)}
    src, dst = torch.tensor(fe.src, device=DEV), torch.tensor(fe.dst, device=DEV)
    ms = load_models(fe, DEV, False)
    E2 = len(fe.src) // 2                                        # first half: one orientation of every link
    cap = fe.caps[:E2].astype(float)
    mon = month_index(fe.H)
    R = dat.R                                                    # (N, H) MW
    lo, hi = np.full(R.shape, np.nan), np.full(R.shape, np.nan)
    for mo in np.unique(mon):
        sel = mon == mo
        blockR = np.where(fe.disp_obs_ok[:, sel], R[:, sel], np.nan)
        lo[:, sel] = np.nanmin(blockR, 1, keepdims=True)
        hi[:, sel] = np.nanmax(blockR, 1, keepdims=True)
    out = {"_meta": {"delta_MW": DELTA, "models": len(ms), "links_checked": int(E2),
                     "capacity_proxy": "99th percentile of |flow| over the sample (fe.caps)",
                     "zone_range": "min and max of observed dispatchable output in the same calendar month"}}
    for b, (x, m) in LINKS.items():
        f = OUTD / f"charged_{b.replace('->', '_')}_graph.npz"
        if not f.exists():
            continue
        hours = np.load(f)["hours"]
        xi, mi = idx[x], idx[m]
        lmask = torch.ones(len(fe.src), device=DEV)
        for e, (s_, d_) in enumerate(zip(fe.src, fe.dst)):
            if {s_, d_} == {xi, mi}:
                lmask[e] = 0.0
        focal = np.array([{s_, d_} == {xi, mi} for s_, d_ in zip(fe.src[:E2], fe.dst[:E2])])
        F0 = dat.Fe[:E2][:, hours].T                             # (T, E2) signed src -> dst, MW
        fok = (dat.Fok[:E2][:, hours] & dat.avail[:E2][:, hours]).T
        res_m, flags_cross, flags_zone = [], [], []
        for mm in ms:
            dRs, dFs = [], []
            with torch.no_grad():
                for i in range(0, len(hours), 128):
                    hs = hours[i:i + 128]
                    bb = dat.batch(hs, DEV)
                    k, xb, g = mm.params(*bb[:4], src, dst, bb[11])
                    g = g * lmask
                    dN = torch.zeros(len(hs), fe.N, device=DEV)
                    dN[:, mi] += DELTA
                    dN[:, xi] -= DELTA / ETA[b]
                    _, dR, dF = solve(k, xb, g, src, dst, dN, fe.N)
                    dRs.append(dR.cpu().numpy())
                    dFs.append(dF[:, :E2].cpu().numpy())
            dR, dF = np.concatenate(dRs), np.concatenate(dFs)    # (T, N), (T, E2)
            F1 = F0 + dF
            use = fok & ~focal[None]
            grows = np.abs(F1) > np.abs(F0)
            lviol = use & grows & (np.abs(F1) > cap[None])
            over = np.where(lviol, np.abs(F1) - cap[None], 0.0)
            near = use & grows & ((cap[None] - np.abs(F0)) < DELTA)
            absdF = np.abs(np.where(use, dF, 0.0))
            R0 = R[:, hours].T
            R1 = R0 + dR
            zok = fe.disp_obs_ok[:, hours].T
            zhi = zok & (dR > 0) & (R1 > hi[:, hours].T + 1e-6)
            zlo = zok & (dR < 0) & (R1 < lo[:, hours].T - 1e-6)
            zneg = zok & (dR < 0) & (R1 < 0)
            zexc = np.where(zhi, R1 - hi[:, hours].T, 0.0) + np.where(zlo, lo[:, hours].T - R1, 0.0)
            # attributable to the perturbation: crossings of the proxy, and MW beyond max(|F0|, proxy)
            cross = use & grows & (np.abs(F0) <= cap[None]) & (np.abs(F1) > cap[None])
            beyond = np.where(use & grows, np.maximum(np.abs(F1) - np.maximum(np.abs(F0), cap[None]), 0.0), 0.0)
            sat = use & grows & (np.abs(F0) >= cap[None])
            absdR = np.abs(np.where(zok, dR, 0.0))
            flags_cross.append(cross.any(1))
            flags_zone.append((zhi | zlo).any(1))
            res_m.append({"hours_any_new_crossing": float(cross.any(1).mean()),
                          "hours_any_link_base_beyond_proxy": float((use & (np.abs(F0) > cap[None])).any(1).mean()),
                          "share_abs_flow_change_beyond_proxy": float(beyond.sum() / max(absdF.sum(), 1e-9)),
                          "share_abs_flow_change_on_links_at_or_above_proxy": float((absdF * sat).sum() / max(absdF.sum(), 1e-9)),
                          "share_abs_output_change_outside_month_range": float(zexc.sum() / max(absdR.sum(), 1e-9)),
                          "hours_any_link_violation": float(lviol.any(1).mean()),
                          "pairs_link_violation": float(lviol.sum() / max(use.sum(), 1)),
                          "max_link_overshoot_MW": float(over.max()),
                          "mean_link_overshoot_MW_when_violated": float(over[lviol].mean()) if lviol.any() else 0.0,
                          "share_abs_flow_change_on_links_within_100MW": float((absdF * near).sum() / max(absdF.sum(), 1e-9)),
                          "hours_any_zone_outside_month_range": float((zhi | zlo).any(1).mean()),
                          "max_zone_excess_MW": float(zexc.max()),
                          "mean_zone_excess_MW_when_outside": float(zexc[zhi | zlo].mean()) if (zhi | zlo).any() else 0.0,
                          "hours_any_zone_negative": float(zneg.any(1).mean()),
                          "max_abs_link_change_MW": float(np.abs(dF[:, ~focal]).max()),
                          "max_abs_zone_change_MW": float(np.abs(dR).max())})
        keys = res_m[0].keys()
        out[b] = {"hours": int(len(hours)),
                  "by_model": res_m,
                  "range_over_models": {k: [min(r[k] for r in res_m), max(r[k] for r in res_m)] for k in keys}}
        # A64: the charged-period decomposition (as in net_summary.py) on all hours and on the hours without flags
        z = np.load(f)
        rho, r_, flow = z["rho"].astype(np.float64), z["r"].astype(np.float64), z["flow"].astype(np.float64)
        a_, b_ = SPLITS["charged"]
        per = (hours >= hour_of(a_)) & (hours < hour_of(b_))
        third = [j for j in range(fe.N) if j not in (xi, mi)]
        dec = {}
        for sname in ("all", "no_new_crossing", "no_new_crossing_no_zone_exit"):
            vals = {k: [] for k in ("absorb_importer", "absorb_exporter", "E_x", "E_m", "third", "network")}
            shares = []
            for s_ in range(len(ms)):
                keep = per.copy()
                if sname != "all":
                    keep &= ~flags_cross[s_]
                if sname == "no_new_crossing_no_zone_exit":
                    keep &= ~flags_zone[s_]
                shares.append(float(keep.sum() / per.sum()))
                w = flow[keep] / flow[keep].sum()
                vals["absorb_importer"].append(float(rho[s_, keep, mi] @ w))
                vals["absorb_exporter"].append(float(-rho[s_, keep, xi] @ w))
                vals["E_x"].append(float(-r_[s_, keep, xi] @ w))
                vals["E_m"].append(float(r_[s_, keep, mi] @ w))
                vals["third"].append(float(r_[s_, keep][:, third].sum(-1) @ w))
                vals["network"].append(float(r_[s_, keep].sum(-1) @ w))
            dec[sname] = {k: [float(np.mean(v)), float(np.min(v)), float(np.max(v))] for k, v in vals.items()}
            dec[sname]["share_of_hours"] = [float(np.mean(shares)), float(np.min(shares)), float(np.max(shares))]
        out[b]["decomposition_charged"] = dec
        print(b, {s: {k: round(v[0], 3) for k, v in d.items()} for s, d in dec.items()}, flush=True)
        print(b, {k: [round(v, 4) for v in out[b]["range_over_models"][k]] for k in keys}, flush=True)
    (OUTD / "net_feasibility.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
