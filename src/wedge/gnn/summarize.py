"""Summaries of the graph-model marginal responses (A49) for the paper and figures.

Per border and period: volume-weighted mean over export hours of E_x, e_out = E_x * eta (comparable with the
redispatch model's e_out and the regression's export coefficient), E_m, kappa = E_x - E_m and the third-zone change,
for every draw (seed x MC dropout); the summary gives the median across draws and the 5-95% range across draws.
Also the share of volume on which the declared factor exceeds the whole draw interval of kappa, and the share with a
robust sign of kappa across draws.
Output: data/processed/gnn_marginal_summary.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import CERT_Q, SPEC, SPLITS, Features, hour_of                    # noqa: E402
from wedge.gnn.marginal import BORDERS                            # noqa: E402

E_DEC = {"GB->NL": 0.430, "GB->BE": 0.430, "RS->HU": 1.041}


OUTD = "data/processed/gnn" if SPEC == "v1" else f"data/processed/gnn/{SPEC}"


def main(suffix=""):
    out = {}
    fe = Features()
    idx = {n: i for i, n in enumerate(fe.nodes)}
    cert = np.zeros(fe.H)
    from datetime import datetime, timezone
    for h in range(hour_of("2026-01-01"), fe.H):
        dt = datetime.fromtimestamp(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() + 3600 * h, tz=timezone.utc)
        cert[h] = CERT_Q.get((dt.year, (dt.month - 1) // 3 + 1), 75.28)
    for b, (x, m, eta) in BORDERS.items():
        f = pathlib.Path(f"{OUTD}/marginal_{b.replace('->', '_')}{suffix}.npz")
        if not f.exists():
            continue
        d = np.load(f)
        hours, draws, flow = d["hours"], d["draws"], d["flow"]
        dnode = d["dnode"].astype(np.float32) if "dnode" in d.files else None       # (D, T, N)
        dcls = d["dcls"].astype(np.float32) if "dcls" in d.files else None          # (D, T, 2, K)
        xi, mi = idx[x], idx[m]
        others = [j for j in range(fe.N) if j not in (xi, mi)]
        out[b] = {}
        for p in ("y2025", "charged", "charged_q3"):
            s, e = (hour_of(v) for v in SPLITS[p])
            sel = (hours >= s) & (hours < e)
            if sel.sum() == 0:
                continue
            w = flow[sel] / flow[sel].sum()
            dr = draws[:, sel]                                            # (D, T, 3)
            ex, em, th = dr[..., 0], dr[..., 1], dr[..., 2]
            kap = ex - em
            q = lambda a: dict(median=float(np.median(a @ w)), p05_p95=[float(np.percentile(a @ w, 5)),
                                                                       float(np.percentile(a @ w, 95))])  # noqa: E731
            robust_pos = (kap.min(0) > 0)
            robust_neg = (kap.max(0) < 0)
            extra = {}
            if dnode is not None:
                h = np.median(dnode[:, sel][:, :, others], 0)                                # (T, J) t per MWh
                s = cert[hours[sel]] if p != "y2025" else fe.carbon[mi, hours[sel]]
                pj = fe.carbon[others][:, hours[sel]].T                                      # (T, J)
                extra["third_abs_sum"] = float(w @ np.abs(h).sum(1))
                extra["third_net"] = float(w @ h.sum(1))
                extra["third_money_bound_eur_mwh"] = float(w @ (np.abs(s[:, None] - pj) * np.abs(h)).sum(1))
                extra["third_money_net_eur_mwh"] = float(w @ ((s[:, None] - pj) * h).sum(1))
            if dcls is not None:
                c = np.median(dcls[:, sel], 0)                                               # (T, 2, K) MW per MW
                # DISP = lignite, coal, gas, oil, hydro_res, pumped; exporter side sign: output falls
                extra["exporter_class_response"] = {k: float(w @ -c[:, 0, j]) for j, k in
                                                    enumerate(["lignite", "coal", "gas", "oil", "hydro_res", "pumped"])}
                extra["importer_class_response"] = {k: float(w @ c[:, 1, j]) for j, k in
                                                    enumerate(["lignite", "coal", "gas", "oil", "hydro_res", "pumped"])}
            out[b][p] = dict(**extra, hours=int(sel.sum()), twh=float(flow[sel].sum() / 1e6),
                             E_x=q(ex), e_out=q(ex * eta), E_m=q(em), kappa=q(kap), third=q(th),
                             declared_above_kappa_share=float(w @ (kap.max(0) < E_DEC[b])),
                             robust_sign_share=float(w @ (robust_pos | robust_neg)),
                             robust_import_raises_share=float(w @ robust_pos),
                             robust_import_lowers_share=float(w @ robust_neg))
            print(b, p, {k: (v if not isinstance(v, dict) else round(v.get("median", 0), 3)) for k, v in out[b][p].items() if "class" not in k})
    name = "gnn_marginal_summary" + suffix + ".json"
    pathlib.Path("data/processed" if SPEC == "v1" else OUTD, name).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
