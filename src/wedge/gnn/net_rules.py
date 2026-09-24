"""Rule regrets against the network reference charge (A53).

Network benchmark: for the removal of one delivered MWh on x -> m in hour t, with signed zone
emission responses r_i (netresp charged output, per seed) and zone carbon costs p_i,
    tau = - sum_i (s - p_i) r_i ,   u = max(0, tau)   (median over seeds).
Also reported: the two-zone benchmark from the same responses (third zones dropped), the common-EU-price
decomposition (exporter term, EU-set term, other zones) and the regret of each rule |a - u|, volume weighted,
January-June 2026 at s = certificate price (and 2025 at s = EUA for reference).
Rules: gross default, all-sources proxy, default with contemporaneous credit, all-sources with credit,
default with annual credit (GB: 81.25 EUR/t), zero charge.
Usage: WEDGE_GNN_SPEC=v4 NETRESP_TAG=r2 python src/wedge/gnn/net_rules.py
Output: <netresp OUTD>/net_rules.json
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import CERT_Q, SPLITS, Features, hour_of      # noqa: E402
from wedge.gnn.netresp import OUTD                                    # noqa: E402

E_DEC = {"GB->NL": 0.430, "GB->BE": 0.430, "RS->HU": 1.041}
E_ALL = {"GB->NL": 0.193, "GB->BE": 0.193, "RS->HU": 0.729}
Q_ANN = {"GB->NL": 81.25, "GB->BE": 81.25, "RS->HU": None}
EU = {"AT", "BE", "CZ", "DE", "DK", "ES", "FR", "HU", "IT", "NL", "PL", "SE", "SK", "RO", "BG", "HR", "SI", "GR",
      "IE", "PT", "NO"}                               # NO applies the EU ETS through the EEA agreement


def cert(hours):
    out = []
    for h in hours:
        dt = datetime.fromtimestamp(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() + 3600 * int(h), tz=timezone.utc)
        out.append(CERT_Q.get((dt.year, (dt.month - 1) // 3 + 1), np.nan))
    return np.array(out)


def main():
    fe = Features()
    idx = {n: i for i, n in enumerate(fe.nodes)}
    res = {}
    for b in E_DEC:
        x, m = b.split("->")
        xi, mi = idx[x], idx[m]
        for variant in ("graph", "local"):
            f = OUTD / f"charged_{b.replace('->', '_')}_{variant}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            hours, r, flow = z["hours"], z["r"], z["flow"]                  # r: (S, T, N) signed, per delivered MWh removed
            p = fe.carbon[:, hours].T                                          # (T, N)
            out = {}
            for per, (a_, b_) in (("2026H1", SPLITS["charged"]), ("2025", SPLITS["y2025"])):
                sel = (hours >= hour_of(a_)) & (hours < hour_of(b_))
                if sel.sum() == 0:
                    continue
                w = flow[sel] / flow[sel].sum()
                s = cert(hours[sel]) if per == "2026H1" else p[sel, mi]
                s = np.where(np.isfinite(s), s, p[sel, mi])
                ps, rs = p[sel], r[:, sel]                                     # (T, N), (S, T, N)
                tau_net = -((s[:, None] - ps)[None] * rs).sum(-1)              # (S, T)
                oth = [j for j in range(fe.N) if j not in (xi, mi)]
                tau_two = -((s[:, None] - ps)[None][..., [xi, mi]] * rs[..., [xi, mi]]).sum(-1)
                u = np.median(np.maximum(tau_net, 0), 0)                        # benchmark: median over seeds
                u2 = np.median(np.maximum(tau_two, 0), 0)
                eu = [idx[n] for n in EU if n in idx and idx[n] != xi]
                non = [j for j in oth if j not in eu]
                # A56 (B5): seed means, so the three terms add up to the reported mean tau
                term_x = np.mean(-(s - ps[:, xi])[None] * rs[..., xi], 0)
                term_eu = np.mean(-((s[:, None] - ps[:, eu])[None] * rs[..., eu]).sum(-1), 0)
                term_non = np.mean(-((s[:, None] - ps[:, non])[None] * rs[..., non]).sum(-1), 0) if non else 0 * u
                px = ps[:, xi]
                rules = {"gross_default": s * E_DEC[b], "all_sources": s * E_ALL[b],
                         "default_credit": np.maximum((s - px) * E_DEC[b], 0),
                         "all_sources_credit": np.maximum((s - px) * E_ALL[b], 0),
                         "zero": np.zeros_like(s)}
                if Q_ANN[b]:
                    rules["default_annual_credit"] = np.maximum((s - Q_ANN[b]) * E_DEC[b], 0)
                o = {"hours": int(sel.sum()), "mean_u_network": float(w @ u), "mean_u_two_zone": float(w @ u2),
                     "mean_tau_network": float(w @ np.mean(tau_net, 0)),
                     "term_exporter": float(w @ term_x), "term_EU_set": float(w @ term_eu),
                     "term_other_zones": float(w @ term_non),
                     "share_positive_u": float(w @ (u > 0))}
                for k, a in rules.items():
                    o[f"regret_{k}"] = float(w @ np.abs(a - u))
                    o[f"regret_two_zone_{k}"] = float(w @ np.abs(a - u2))
                out[per] = o
                print(b, variant, per, {k: round(v, 3) for k, v in o.items()}, flush=True)
            res[f"{b}|{variant}"] = out
    (OUTD / "net_rules.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
