"""Charged-import retention of the single-link intervention, and the rule comparison on the net charge change (A66).

The network responses remove one delivered MWh on the charged link x -> m with every other link available. A rule sets
one charge a per MWh for every import into the EU from the exporter x, so energy that reaches EU zones over x's other
charged links pays the same charge. Along the response path the charged imports of x fall by
    b(t) = 1 - sum_{e in T(x), e != l} dQ_e(t) / delta,   dQ_e = max(s_e (F_e + dF_e), 0) - max(s_e F_e, 0),
per MWh removed, with s_e orienting link e from x into the EU zone, F_e the observed flow and dF_e the model's flow
change (the linear change s_e dF_e where the flow is unobserved). The net charge change along the path is H(a) = a b,
and the path-consistent regret is |a b - u|; the single-link regret |a - u| of net_rules/net_states is the case b = 1.
T(GB) = the links to NL, BE, FR, DK and IE. The graph aggregates parallel cables, and the charge
applies to the positive imports of each cable (simultaneous exports are not netted), so on the aggregated FR edge (IFA,
IFA2, ElecLink) and IE edge (East-West, Greenlink, and Moyle into Northern Ireland, not an EU import) the model's edge
change is allocated to the cables in proportion to capacity and the positive part is taken per cable, with the observed
hourly cable flows (Elexon INTOUTHH). Bounds: the largest and the smallest charged change over all allocations of the
edge change to its cables (chi_low, chi_high); the earlier edge-level form is kept as chi_aggregate.
T(RS) = the links to HU, RO, BG and HR. For RS the change in charged imports of the other charged exporters of the graph
(BA -> HR, ME -> IT, MK -> BG and GR) per MWh removed is also reported; their charges follow their own defaults.
States and rules as in net_states.py (GB: S26, W27, NOCPS, CONV; RS: S26), January-June 2026, network population.
The per-hour chi of the whole network population (2025 to August 2026) is saved for net_frontier.py.
Usage: WEDGE_GNN_SPEC=v4 NETRESP_TAG=r2 python src/wedge/gnn/net_taxbase.py
Output: <netresp OUTD>/net_taxbase.json and taxbase_<border>_<variant>.npz (hours; chi, chi_aggregate, chi_low,
chi_high: models x hours)
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import SPLITS, Features, hour_of                        # noqa: E402
from wedge.gnn.net_rules import E_ALL, E_DEC, cert                               # noqa: E402
from wedge.gnn.net_states import Q_UKA, Q_WITH_CPS, monthly                      # noqa: E402
from wedge.gnn.netresp import ETA, OUTD, Data, load_models, solve               # noqa: E402
from wedge.gnn.outage_study import LINKS                                        # noqa: E402

DEV = "cuda"
DELTA = 100.0
CHARGED = {"GB": ("NL", "BE", "FR", "DK", "IE"), "RS": ("HU", "RO", "BG", "HR")}
OTHER_EXPORTERS = {"BA": ("HR",), "ME": ("IT",), "MK": ("BG", "GR")}
RAWGB = pathlib.Path("data/raw/gnn/gb")
# GB cables of the aggregated FR and IE edges: capacity (MW) and whether flow into the neighbour is an EU import
CABLES = {"FR": {"France(IFA)": (2000.0, True), "IFA2 (INTIFA2)": (1000.0, True), "Eleclink (INTELEC)": (1000.0, True)},
          "IE": {"Ireland(East-West)": (500.0, True), "Ireland (Greenlink)": (504.0, True),
                 "Northern Ireland(Moyle)": (500.0, False)}}
CAB = {}


def gb_cable_flows(H):
    """Hourly flow GB -> neighbour (MW) of every cable in CABLES, the mean of the half-hourly Elexon outturn."""
    t0 = datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp()
    names = {n for c in CABLES.values() for n in c}
    s, c, seen = {n: np.zeros(H) for n in names}, {n: np.zeros(H) for n in names}, set()
    for f in sorted(glob.glob(str(RAWGB / "202[4-6]" / "gb_ic_*.json"))):
        for r in json.load(open(f, encoding="utf8")).get("data", []):
            n = r.get("interconnectorName")
            if n not in names or r.get("generation") is None or (r.get("startTime"), n) in seen:
                continue
            seen.add((r.get("startTime"), n))
            h = int((datetime.fromisoformat(r["startTime"].replace("Z", "+00:00")).timestamp() - t0) // 3600)
            if 0 <= h < H:
                s[n][h] -= float(r["generation"])                                   # Elexon: + = into GB
                c[n][h] += 1
    return {n: np.where(c[n] > 0, s[n] / np.maximum(c[n], 1), np.nan) for n in names}


def dq_cables(j, Fc, dF, mode):
    """Change in the charged flow (MW) over the cables of the aggregated GB edge to j. Fc {cable: (T,) flow GB -> j},
    dF (T,) model change of the edge flow GB -> j. mode 'cap': dF split by capacity, positive part per cable;
    'hi' / 'lo': the largest / smallest charged change over all splits of dF across the cables."""
    cab = CABLES[j]
    if mode == "cap":
        tot = sum(cap for cap, _ in cab.values())
        out = np.zeros_like(dF)
        for n, (cap, charged) in cab.items():
            if charged:
                d = dF * cap / tot
                out += np.where(np.isfinite(Fc[n]), np.maximum(Fc[n] + d, 0) - np.maximum(Fc[n], 0), d)
        return out
    if mode == "hi":                       # convex in the split: the maximum puts dF on one cable
        single = [np.where(np.isfinite(Fc[n]), np.maximum(Fc[n] + dF, 0) - np.maximum(Fc[n], 0), dF) if charged
                  else np.zeros_like(dF) for n, (cap, charged) in cab.items()]
        return np.max(np.stack(single), 0)
    imp_room = sum(np.nan_to_num(np.maximum(-Fc[n], 0)) for n, (cap, ch) in cab.items() if ch)
    exp_room = sum(np.where(np.isfinite(Fc[n]), np.maximum(Fc[n], 0), np.inf) for n, (cap, ch) in cab.items() if ch)
    free = any(not ch for cap, ch in cab.values())                               # an uncharged cable absorbs dF > 0
    lo_pos = np.zeros_like(dF) if free else np.maximum(dF - imp_room, 0)
    lo_neg = -np.minimum(-dF, exp_room)
    return np.where(dF > 0, lo_pos, lo_neg)


def wm(w, a):
    return float(w @ a)


def link_sign(fe, E2, a, b_):
    """Index of the undirected link a-b_ among the first E2 edges and the sign that orients it a -> b_."""
    ia, ib = fe.nodes.index(a), fe.nodes.index(b_)
    for e, (s_, d_) in enumerate(zip(fe.src[:E2], fe.dst[:E2])):
        if (s_, d_) == (ia, ib):
            return e, 1.0
        if (s_, d_) == (ib, ia):
            return e, -1.0
    return None, None


def dq(F0, ok, dF, sgn):
    """Change in the charged (x -> EU) flow of one link: F0, ok, dF (T,), in MW."""
    lin = sgn * dF
    nonlin = np.maximum(sgn * (F0 + dF), 0.0) - np.maximum(sgn * F0, 0.0)
    return np.where(ok, nonlin, lin)


def rules_for(b, s, px, extra):
    out = {"gross_default": s * E_DEC[b], "all_sources": s * E_ALL[b],
           "default_credit": np.maximum((s - px) * E_DEC[b], 0), "all_sources_credit": np.maximum((s - px) * E_ALL[b], 0),
           "zero": np.zeros_like(s)}
    for k, q in extra.items():
        out[k] = np.maximum((s - q) * E_DEC[b], 0)
    return out


def main():
    fe = Features()
    dat = Data(fe)
    idx = {n: i for i, n in enumerate(fe.nodes)}
    src, dst = torch.tensor(fe.src, device=DEV), torch.tensor(fe.dst, device=DEV)
    E2 = len(fe.src) // 2
    a_, b_ = SPLITS["charged"]
    CAB.update(gb_cable_flows(fe.H))
    out = {"_meta": {"delta_MW": DELTA, "charged_links": {k: list(v) for k, v in CHARGED.items()},
                     "other_exporters": {k: list(v) for k, v in OTHER_EXPORTERS.items()},
                     "period": "network population, January-June 2026", "tag": str(OUTD)}}
    for b, (x, m) in LINKS.items():
        xi, mi = idx[x], idx[m]
        for variant, local in (("graph", False), ("local", True)):
            f = OUTD / f"charged_{b.replace('->', '_')}_{variant}.npz"
            if not f.exists():
                continue
            try:
                ms = load_models(fe, DEV, local)
            except FileNotFoundError:
                continue
            z = np.load(f)
            hours_all = z["hours"]
            sel = (hours_all >= hour_of(a_)) & (hours_all < hour_of(b_))
            hours = hours_all                                                   # solve on the whole population
            rho_saved, r = z["rho"].astype(np.float64), z["r"][:, sel].astype(np.float64)
            flow = z["flow"][sel].astype(np.float64)
            w = flow / flow.sum()
            lmask = torch.ones(len(fe.src), device=DEV)
            for e, (s_, d_) in enumerate(zip(fe.src, fe.dst)):
                if {s_, d_} == {xi, mi}:
                    lmask[e] = 0.0
            F0 = dat.Fe[:E2][:, hours].T.astype(np.float64)                     # (T, E2) signed src -> dst
            fok = (dat.Fok[:E2][:, hours] & dat.avail[:E2][:, hours]).T
            chan = {j: link_sign(fe, E2, x, j) for j in CHARGED[x] if j != m}
            chan = {j: v for j, v in chan.items() if v[0] is not None}
            others = {f"{o}->{j}": link_sign(fe, E2, o, j) for o, js in OTHER_EXPORTERS.items() for j in js}
            others = {k: v for k, v in others.items() if v[0] is not None}
            untaxed = {j: link_sign(fe, E2, x, fe.nodes[j]) for j in range(fe.N)
                       if fe.nodes[j] not in CHARGED[x] and j != xi}
            untaxed = {fe.nodes[j]: v for j, v in untaxed.items() if v[0] is not None}
            cab = {j: {n: CAB[n][hours] for n in CABLES[j]} for j in CABLES} if x == "GB" else {}
            bs, per_link, per_other, per_untaxed, drho = [], [], [], [], []
            bs_agg, bs_lo, bs_hi = [], [], []
            for si, mm in enumerate(ms):
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
                        dRs.append(dR.double().cpu().numpy())
                        dFs.append(dF[:, :E2].double().cpu().numpy())
                dR, dF = np.concatenate(dRs), np.concatenate(dFs)
                # the saved responses are dR / delta per removed MWh (rho, netresp.intervene); check the re-solve
                drho.append(float(np.abs(dR / DELTA - rho_saved[si]).max()))
                q_agg = {j: dq(F0[:, e], fok[:, e], dF[:, e], sg) / DELTA for j, (e, sg) in chan.items()}
                q, q_hi, q_lo = dict(q_agg), dict(q_agg), dict(q_agg)
                for j, (e, sg) in chan.items():
                    if j in cab:
                        q[j] = dq_cables(j, cab[j], sg * dF[:, e], "cap") / DELTA
                        q_hi[j] = dq_cables(j, cab[j], sg * dF[:, e], "hi") / DELTA
                        q_lo[j] = dq_cables(j, cab[j], sg * dF[:, e], "lo") / DELTA
                bs.append(1.0 - sum(q.values()))
                bs_agg.append(1.0 - sum(q_agg.values()))
                bs_lo.append(1.0 - sum(q_hi.values()))                           # most charged rerouting
                bs_hi.append(1.0 - sum(q_lo.values()))
                per_link.append({j: wm(w, v[sel]) for j, v in q.items()})
                per_other.append({k: wm(w, (dq(F0[:, e], fok[:, e], dF[:, e], sg) / DELTA)[sel])
                                  for k, (e, sg) in others.items()})
                per_untaxed.append({j: wm(w, (sg * dF[:, e] / DELTA)[sel]) for j, (e, sg) in untaxed.items()})
            bs_all = np.stack(bs)                                               # (S, T_all)
            alt_all = {"aggregate": np.stack(bs_agg), "low": np.stack(bs_lo), "high": np.stack(bs_hi)}
            assert np.all(alt_all["low"] <= bs_all + 1e-12) and np.all(bs_all <= alt_all["high"] + 1e-12)
            np.savez_compressed(OUTD / f"taxbase_{b.replace('->', '_')}_{variant}.npz", hours=hours_all, chi=bs_all,
                                chi_aggregate=alt_all["aggregate"], chi_low=alt_all["low"], chi_high=alt_all["high"])
            bs, hours = bs_all[:, sel], hours_all[sel]                          # January-June 2026
            alt = {k: v[:, sel] for k, v in alt_all.items()}
            o = {"hours": int(len(hours)), "max_abs_rho_check": max(drho),
                 "b_mean_by_model": [wm(w, bi) for bi in bs],
                 "b_share_hours_below_0.67": [wm(w, (bi < 0.67).astype(float)) for bi in bs],
                 "b_quantiles_median_model": [float(v) for v in np.quantile(np.median(bs, 0), [0.1, 0.5, 0.9])],
                 "charged_other_links_dQ_per_MWh": per_link, "untaxed_links_dF_per_MWh": per_untaxed,
                 "other_exporters_charged_dQ_per_MWh": per_other}
            for k, v in alt.items():
                o[f"chi_mean_by_model_{k}"] = [wm(w, bi) for bi in v]
            # rule comparison: single-link |a - u| and path-consistent |a b - u|, per model, and the median-model form
            p0 = fe.carbon[:, hours].T.astype(np.float64)
            pi = cert(hours)
            pi = np.where(np.isfinite(pi), pi, p0[:, mi])
            eua = monthly(hours, "eua_eur")
            if x == "GB":
                uka = monthly(hours, "uka_eur")
                states = {"S26": (p0[:, xi], pi, {"default_annual_credit_81": Q_WITH_CPS[0]}),
                          "W27": (p0[:, xi], eua, {"default_annual_credit_81": Q_WITH_CPS[0]}),
                          "NOCPS": (uka, pi, {"default_annual_credit_uka": Q_UKA}),
                          "CONV": (eua, eua, {})}
            else:
                states = {"S26": (p0[:, xi], pi, {})}
            o["states"] = {}
            for st, (px, s, extra) in states.items():
                p = p0.copy()
                p[:, xi] = px
                tau = -((s[:, None] - p)[None] * r).sum(-1)                     # (S, T)
                u_s = np.maximum(tau, 0)
                u = np.median(u_s, 0)
                bmed = np.median(bs, 0)
                ev = {"mean_u": wm(w, u), "single": {}, "single_range": {}, "path": {}, "path_range": {},
                      "path_mean_charge_change": {}, "path_aggregate": {}, "path_chi_low": {}, "path_chi_high": {}}
                amed = {k: np.median(v, 0) for k, v in alt.items()}
                for k, a in rules_for(b, s, px, extra).items():
                    ev["single"][k] = wm(w, np.abs(a - u))
                    ev["single_range"][k] = [min(wm(w, np.abs(a - u_s[i])) for i in range(len(ms))),
                                             max(wm(w, np.abs(a - u_s[i])) for i in range(len(ms)))]
                    ev["path"][k] = wm(w, np.abs(a * bmed - u))
                    ev["path_range"][k] = [min(wm(w, np.abs(a * bs[i] - u_s[i])) for i in range(len(ms))),
                                           max(wm(w, np.abs(a * bs[i] - u_s[i])) for i in range(len(ms)))]
                    ev["path_mean_charge_change"][k] = wm(w, a * bmed)
                    for kk, key in (("aggregate", "path_aggregate"), ("low", "path_chi_low"), ("high", "path_chi_high")):
                        ev[key][k] = wm(w, np.abs(a * amed[kk] - u))
                o["states"][st] = ev
                print(b, variant, st, "b", round(wm(w, bmed), 3), "u", round(ev["mean_u"], 2),
                      {k: (round(ev["single"][k], 2), round(ev["path"][k], 2)) for k in ev["single"]}, flush=True)
            if x == "GB" and variant == "graph":
                # outage-anchored rows of the robustness check (net_states.py): a constant GB emission response from
                # the pooled outage estimates, benchmark (pi - p_x) E_x, charge change a * chi with this model's chi
                pool = json.loads(pathlib.Path("data/processed/gnn/outage_pooled_gb_v4.json").read_text(encoding="utf-8"))
                bmed = np.median(bs, 0)
                o["outage_anchored_path"] = {}
                for name, ex in (("outage_A", -pool["rows"]["E_GB"]["A_block"]["did"][2]),
                                 ("outage_B", -pool["rows"]["E_GB"]["B_block_hour"]["did"][2])):
                    o["outage_anchored_path"][name] = {"E_x": ex}
                    for st, pxs in (("S26", p0[:, xi]), ("NOCPS", monthly(hours, "uka_eur"))):
                        u = np.maximum((pi - pxs) * ex, 0)
                        rl = rules_for(b, pi, pxs, {})
                        o["outage_anchored_path"][name][st] = {
                            "single": {k: wm(w, np.abs(a - u)) for k, a in rl.items()},
                            "path": {k: wm(w, np.abs(a * bmed - u)) for k, a in rl.items()}}
                    print(b, name, "NOCPS", {k: (round(v, 2), round(o["outage_anchored_path"][name]["NOCPS"]["path"][k], 2))
                                             for k, v in o["outage_anchored_path"][name]["NOCPS"]["single"].items()}, flush=True)
            print(b, variant, "b by model", [round(v, 3) for v in o["b_mean_by_model"]], "rho check", max(drho),
                  "charged dQ", {j: round(np.mean([pl[j] for pl in per_link]), 3) for j in per_link[0]},
                  "untaxed dF", {j: round(np.mean([pl[j] for pl in per_untaxed]), 3) for j in per_untaxed[0]},
                  "other exporters", {j: round(np.mean([pl[j] for pl in per_other]), 3) for j in per_other[0]} if per_other[0] else {},
                  flush=True)
            out[f"{b}|{variant}"] = o
    (OUTD / "net_taxbase.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def summary():
    """Quantities quoted in the text, from the saved outputs (no model run): per border and the graph variant, the
    exporter response E_x, chi (median over models, volume-weighted over January-June 2026), E_x per MWh of charged
    imports, the path crossover chi (e_dec + e_all) / 2 of the two credited factors, and the rerouting by link group."""
    fe = Features()
    idx = {n: i for i, n in enumerate(fe.nodes)}
    res = json.loads((OUTD / "net_taxbase.json").read_text(encoding="utf-8"))
    a_, b_ = SPLITS["charged"]
    sm = {}
    for b, (x, m) in LINKS.items():
        key = f"{b}|graph"
        if key not in res:
            continue
        z = np.load(OUTD / f"charged_{b.replace('->', '_')}_graph.npz")
        tb = np.load(OUTD / f"taxbase_{b.replace('->', '_')}_graph.npz")
        assert np.array_equal(z["hours"], tb["hours"])
        h = z["hours"]
        sel = (h >= hour_of(a_)) & (h < hour_of(b_))
        w = z["flow"][sel] / z["flow"][sel].sum()
        ex = w @ (-np.median(z["r"][:, sel, idx[x]], 0))
        chi = w @ np.median(tb["chi"][:, sel], 0)
        alts = {k: float(w @ np.median(tb[f"chi_{k}"][:, sel], 0)) for k in ("aggregate", "low", "high")}
        per = res[key]["charged_other_links_dQ_per_MWh"]
        links = {j: float(np.mean([pl[j] for pl in per])) for j in per[0]}
        sm[b] = {"E_x": float(ex), "chi": float(chi), "E_x_per_charged_MWh": float(ex / chi),
                 "crossover_single": (E_DEC[b] + E_ALL[b]) / 2, "crossover_path": float(chi * (E_DEC[b] + E_ALL[b]) / 2),
                 "rerouted_charged_by_link": links, "rerouted_charged_total": float(sum(links.values())),
                 "chi_aggregate": alts["aggregate"], "chi_low": alts["low"], "chi_high": alts["high"]}
        if x == "GB":
            sm[b]["rerouted_charged_FR_plus_IE"] = links.get("FR", 0.0) + links.get("IE", 0.0)
            # regret range over the splits among the cables: in each hour chi lies in [chi_low, chi_high]
            # (median over models); |a chi - u| is convex in chi, so its hourly maximum is at an end and its minimum
            # at an end or zero where u / a lies inside; the bounds average these hourly extremes
            hs = h[sel]
            r = z["r"][:, sel].astype(np.float64)
            lo, hi = (np.median(tb[f"chi_{k}"][:, sel], 0) for k in ("low", "high"))
            p0 = fe.carbon[:, hs].T.astype(np.float64)
            pi = cert(hs)
            pi = np.where(np.isfinite(pi), pi, p0[:, idx[m]])
            sm[b]["regret_range_over_cable_splits"] = {}
            for st, px in (("S26", p0[:, idx[x]]), ("NOCPS", monthly(hs, "uka_eur"))):
                p = p0.copy()
                p[:, idx[x]] = px
                u = np.median(np.maximum(-((pi[:, None] - p)[None] * r).sum(-1), 0), 0)
                rng = {}
                for k, a in rules_for(b, pi, px, {}).items():
                    e_lo, e_hi = np.abs(a * lo - u), np.abs(a * hi - u)
                    inside = (a > 0) & (u >= a * lo) & (u <= a * hi)
                    rng[k] = [float(w @ np.where(inside, 0.0, np.minimum(e_lo, e_hi))), float(w @ np.maximum(e_lo, e_hi))]
                sm[b]["regret_range_over_cable_splits"][st] = rng
                if st == "S26":
                    # credits for the UK allowance alone in the 2026 state (Table 9 upper block), along the path
                    uka = monthly(hs, "uka_eur")
                    chi = np.median(tb["chi"][:, sel], 0)
                    ao = {"default_credit_uka": np.maximum((pi - uka) * E_DEC[b], 0),
                          "default_annual_credit_uka": np.maximum((pi - Q_UKA) * E_DEC[b], 0)}
                    sm[b]["S26_allowance_only_credit"] = {k: {"single": float(w @ np.abs(a - u)),
                                                              "path": float(w @ np.abs(a * chi - u))}
                                                          for k, a in ao.items()}
            print(b, "NOCPS range over splits", {k: [round(v, 3) for v in rr]
                                                  for k, rr in sm[b]["regret_range_over_cable_splits"]["NOCPS"].items()})
        print(b, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sm[b].items()}, flush=True)
    res["_summary"] = sm
    (OUTD / "net_taxbase.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    summary() if sys.argv[1:] == ["summary"] else main()
