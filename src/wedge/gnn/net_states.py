"""Rule regrets against the network benchmark in the four institutional states, the network form of the three-term
decomposition, and the liability gap (A58).

Network benchmark (eq. tauref-net): for the removal of one delivered MWh on x -> m in hour t, with the signed zone
emission responses r_i of the network-response model (charged_<border>_<variant>.npz, one row per trained model) and
the zone carbon costs p_i,
    tau = - sum_i (s - p_i) r_i ,  u = max(0, tau).
The benchmark is the median over trained models of u; ranges are the min and max over models of the volume-weighted
mean (each model held fixed over all hours). The two-zone accounting comparator uses the same local emission responses
m_i with E_x = m_x / eta and E_m = m_m. Responses are held at the observed 2026 state in every institutional
state (a fixed-state accounting experiment).

States (GB borders; as in setup Sec. s-rules):
  S26    s = certificate price pi_t, p_GB = UKA + CPS (observed)
  W27    matched prices: s = pi_t := monthly EUA (the importer's carbon cost)
  NOCPS  CPS removed from dispatch and from the credit: p_GB = UKA, s = pi_t
  CONV   UK-EU carbon price convergence: p_GB = EUA, s = pi_t := EUA
Rules (eq. rule): gross default, all-sources proxy, default and all-sources with a contemporaneous credit at the GB
dispatch carbon cost of the state, default with an annual credit (81.25 and 91.58 EUR/t with the Support, 60.50
allowance only), default with a contemporaneous allowance-only credit, zero charge. Serbia: S26 only, credit at the
nominal 4 EUR/t.
Decomposition (Proposition 1 with network responses), gross default at s = pi_t:
    pi e_dec - tau = pi (e_dec - E_x) + p_x E_x + sum_{j != x} (pi - p_j) r_j ,
with the last term split into the zones at the EU allowance price and all other zones.
Liability gap: sum over covered hours of volume x (gross default - benchmark), million EUR, with the covered share of
the border's export volume in January-June 2026.
Usage: WEDGE_GNN_SPEC=v4 NETRESP_TAG=r2 python src/wedge/gnn/net_states.py
Output: <netresp OUTD>/net_states.json
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import SPLITS, Features, hour_of      # noqa: E402
from wedge.gnn.net_rules import E_ALL, E_DEC, cert             # noqa: E402
from wedge.gnn.netresp import ETA, OUTD                        # noqa: E402

PRICES = json.loads(pathlib.Path("data/processed/gnn_prices_monthly.json").read_text(encoding="utf-8"))
H1 = [f"2026-0{m}" for m in range(1, 7)]
Q_WITH_CPS = (round(float(np.mean([PRICES[k]["gb_carbon_eur"] for k in H1])), 2),
              round(float(PRICES["2025-12"]["gb_carbon_eur"]), 2))
Q_UKA = round(float(np.mean([PRICES[k]["uka_eur"] for k in H1])), 2)


def month_keys(hours):
    t0 = datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp()
    return [datetime.fromtimestamp(t0 + 3600 * int(h), tz=timezone.utc).strftime("%Y-%m") for h in hours]


def monthly(hours, key):
    return np.array([PRICES[k][key] for k in month_keys(hours)])


def wmean(w, a):
    return float(w @ a)


def evaluate(r, p, s, cert_price, xi, e_dec, e_all, rules_extra, w):
    """r (S,T,N), p (T,N), s and cert_price (T,). Returns benchmark, rule regrets (median-model benchmark and the
    range over models) and the two-zone benchmark."""
    tau = -((s[:, None] - p)[None] * r).sum(-1)                                   # (S, T)
    u_s = np.maximum(tau, 0)
    u = np.median(u_s, 0)
    px = p[:, xi]
    rules = {"gross_default": cert_price * e_dec, "all_sources": cert_price * e_all,
             "default_credit": np.maximum((cert_price - px) * e_dec, 0),
             "all_sources_credit": np.maximum((cert_price - px) * e_all, 0),
             "zero": np.zeros_like(s)}
    for k, q in rules_extra.items():
        rules[k] = np.maximum((cert_price - q) * e_dec, 0)
    out = {"mean_u": wmean(w, u), "mean_u_range": [float(min(w @ u_s.T)), float(max(w @ u_s.T))],
           "mean_tau": wmean(w, tau.mean(0)), "share_positive_u": wmean(w, (u > 0).astype(float)),
           "regret": {}, "regret_range": {}, "mean_charge": {}}
    for k, a in rules.items():
        out["regret"][k] = wmean(w, np.abs(a - u))
        per = [wmean(w, np.abs(a - u_s[i])) for i in range(u_s.shape[0])]
        out["regret_range"][k] = [float(min(per)), float(max(per))]
        out["mean_charge"][k] = wmean(w, a)
    return out, u, rules


def main():
    fe = Features()
    idx = {n: i for i, n in enumerate(fe.nodes)}
    eua_like = [i for i, n in enumerate(fe.nodes) if n not in ("GB", "RS", "BA", "ME", "MK")]
    print("annual credit prices", Q_WITH_CPS, Q_UKA, flush=True)
    res = {"annual_credit_prices": {"with_cps_h1_mean": Q_WITH_CPS[0], "with_cps_dec2025": Q_WITH_CPS[1],
                                    "allowance_only_h1_mean": Q_UKA}}
    a_, b_ = SPLITS["charged"]
    for b in E_DEC:
        x, m = b.split("->")
        xi, mi = idx[x], idx[m]
        for variant in ("graph", "local"):
            f = OUTD / f"charged_{b.replace('->', '_')}_{variant}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            hours, r, flow = z["hours"], z["r"].astype(np.float64), z["flow"].astype(np.float64)
            sel = (hours >= hour_of(a_)) & (hours < hour_of(b_))
            hours, r, flow, ms = hours[sel], r[:, sel], flow[sel], z["mslope"][sel].astype(np.float64)
            w = flow / flow.sum()
            p0 = fe.carbon[:, hours].T.astype(np.float64)                          # (T, N), observed state
            pi = cert(hours)
            pi = np.where(np.isfinite(pi), pi, p0[:, mi])
            eua = monthly(hours, "eua_eur")
            o = {"hours": int(sel.sum()), "volume_gwh": float(flow.sum() / 1e3)}
            if x == "GB":
                uka = monthly(hours, "uka_eur")
                states = {}
                for st in ("S26", "W27", "NOCPS", "CONV"):
                    p = p0.copy()
                    if st == "NOCPS":
                        p[:, xi] = uka
                    if st == "CONV":
                        p[:, xi] = eua
                    s = eua if st in ("W27", "CONV") else pi
                    extra = {}
                    if st in ("S26", "W27"):
                        extra = {"default_annual_credit_81": Q_WITH_CPS[0], "default_annual_credit_92": Q_WITH_CPS[1],
                                 "default_annual_credit_uka": Q_UKA}
                        extra_cont = uka
                    elif st == "NOCPS":
                        extra = {"default_annual_credit_uka": Q_UKA}
                        extra_cont = None
                    else:
                        extra_cont = None
                    ev, u, rules = evaluate(r, p, s, s, xi, E_DEC[b], E_ALL[b], extra, w)
                    if extra_cont is not None:                                     # contemporaneous allowance-only credit
                        a = np.maximum((s - extra_cont) * E_DEC[b], 0)
                        tau = -((s[:, None] - p)[None] * r).sum(-1)
                        u_s = np.maximum(tau, 0)
                        ev["regret"]["default_credit_uka"] = wmean(w, np.abs(a - u))
                        per = [wmean(w, np.abs(a - u_s[i])) for i in range(u_s.shape[0])]
                        ev["regret_range"]["default_credit_uka"] = [float(min(per)), float(max(per))]
                        ev["mean_charge"]["default_credit_uka"] = wmean(w, a)
                    # two-zone accounting comparator: E_x = m_x / eta, E_m = m_m (same local responses)
                    u2 = np.maximum((s - p[:, xi]) * ms[:, xi] / ETA[b] - (s - p[:, mi]) * ms[:, mi], 0)
                    ev["mean_u_two_zone"] = wmean(w, u2)
                    ev["regret_two_zone"] = {k: wmean(w, np.abs(a - u2)) for k, a in rules.items()}
                    states[st] = ev
                    print(b, variant, st, "u", round(ev["mean_u"], 3), {k: round(v, 2) for k, v in ev["regret"].items()},
                          flush=True)
                o["states"] = states
            else:
                ev, u, rules = evaluate(r, p0, pi, pi, xi, E_DEC[b], E_ALL[b], {}, w)
                # upper bound for the energy that leaves the graph (boundary outflow b_t per MWh removed): valued as
                # if it displaced lignite (1.10 t/MWh, the highest class factor) in zones without a carbon price
                rho = z["rho"][:, sel].astype(np.float64)                              # (S, T, N)
                bout = np.maximum(rho.sum(-1) - (1 - 1 / ETA[b]), 0.0)                  # (S, T) leaving the graph
                tau_s = -((pi[:, None] - p0)[None] * r).sum(-1)
                u_up = np.median(np.maximum(tau_s + pi[None] * 1.10 * bout, 0), 0)
                ev["boundary_outflow_mean"] = wmean(w, bout.mean(0))
                ev["mean_u_boundary_upper"] = wmean(w, u_up)
                ev["regret_gross_default_boundary_upper"] = wmean(w, np.abs(pi * E_DEC[b] - u_up))
                ev["regret_all_sources_boundary_upper"] = wmean(w, np.abs(pi * E_ALL[b] - u_up))
                u2 = np.maximum((pi - p0[:, xi]) * ms[:, xi] / ETA[b] - (pi - p0[:, mi]) * ms[:, mi], 0)
                ev["mean_u_two_zone"] = wmean(w, u2)
                ev["regret_two_zone"] = {k: wmean(w, np.abs(a - u2)) for k, a in rules.items()}
                o["states"] = {"S26": ev}
                print(b, variant, "S26 u", round(ev["mean_u"], 3), {k: round(v, 2) for k, v in ev["regret"].items()},
                      flush=True)
            if x == "GB":
                # outage-anchored check: a constant GB emission response from the pooled outage estimates
                # (outage_pooled_gb_v4.json, E_GB, all hours, specifications A and B) and the two-zone value,
                # benchmark (pi - p_x) E_x in the 2026 and post-CPS states
                pool = json.loads(pathlib.Path("data/processed/gnn/outage_pooled_gb_v4.json").read_text(encoding="utf-8"))
                anchors = {"outage_A": -pool["rows"]["E_GB"]["A_block"]["did"][2],
                           "outage_B": -pool["rows"]["E_GB"]["B_block_hour"]["did"][2],
                           "two_zone": float(w @ (ms[:, xi] / ETA[b]))}
                uka = monthly(hours, "uka_eur")
                o["outage_anchored"] = {}
                for name, ex in anchors.items():
                    o["outage_anchored"][name] = {"E_x": ex}
                    for st, pxs in (("S26", p0[:, xi]), ("NOCPS", uka)):
                        u = np.maximum((pi - pxs) * ex, 0)
                        rl = {"gross_default": pi * E_DEC[b], "all_sources": pi * E_ALL[b],
                              "default_credit": np.maximum((pi - pxs) * E_DEC[b], 0),
                              "all_sources_credit": np.maximum((pi - pxs) * E_ALL[b], 0), "zero": 0 * pi}
                        o["outage_anchored"][name][st] = {k: wmean(w, np.abs(a - u)) for k, a in rl.items()}
            # decomposition of the gross default (Proposition 1 with network responses), observed state, s = pi
            Ex = -r[..., xi]                                                        # (S, T)
            px = p0[:, xi]
            dec = pi[None] * (E_DEC[b] - Ex)
            cred = px[None] * Ex
            oth = [j for j in range(fe.N) if j != xi]
            eu = [j for j in oth if j in eua_like]
            non = [j for j in oth if j not in eua_like]
            val_eu = ((pi[:, None] - p0[:, eu])[None] * r[..., eu]).sum(-1)
            val_non = ((pi[:, None] - p0[:, non])[None] * r[..., non]).sum(-1) if non else 0 * dec
            gap = pi[None] * E_DEC[b] + ((pi[:, None] - p0)[None] * r).sum(-1)       # gross - tau, per model and hour
            chk = dec + cred + val_eu + val_non - gap
            assert np.abs(chk).max() < 1e-6, np.abs(chk).max()
            terms = {"declaration_factor": dec, "credit": cred, "valuation_eu_set": val_eu, "valuation_other": val_non}
            o["decomposition"] = {k: {"mean": wmean(w, v.mean(0)), "range": [float(min(w @ v.T)), float(max(w @ v.T))]}
                                  for k, v in terms.items()}
            o["decomposition"]["gap_gross_minus_tau"] = wmean(w, gap.mean(0))
            # the same decomposition under the two-zone accounting (E_x = m_x / eta, E_m = m_m)
            ex2, em2 = ms[:, xi] / ETA[b], ms[:, mi]
            o["decomposition_two_zone"] = {"declaration_factor": wmean(w, pi * (E_DEC[b] - ex2)),
                                           "credit": wmean(w, px * ex2),
                                           "valuation_importer": wmean(w, (pi - p0[:, mi]) * em2)}
            # liability gap on covered volume (MWh x EUR/MWh -> million EUR)
            u_med = np.median(np.maximum(-((pi[:, None] - p0)[None] * r).sum(-1), 0), 0)
            o["liability_gross_meur"] = float(flow @ (pi * E_DEC[b]) / 1e6)
            o["liability_gap_meur"] = float(flow @ (pi * E_DEC[b] - u_med) / 1e6)
            o["benchmark_liability_meur"] = float(flow @ u_med / 1e6)
            # coverage: covered export volume against all export volume of the border in the period
            fall = fe.F[xi, mi, hour_of(a_):hour_of(b_)].astype(np.float64)
            o["covered_share_of_export_volume"] = float(flow.sum() / fall[fall > 0].sum())
            o["export_volume_gwh_all"] = float(fall[fall > 0].sum() / 1e3)
            res[f"{b}|{variant}"] = o
            print(b, variant, "decomp", {k: round(v["mean"], 2) for k, v in o["decomposition"].items()
                                         if isinstance(v, dict)}, "liab gap", round(o["liability_gap_meur"], 1),
                  flush=True)
    (OUTD / "net_states.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
