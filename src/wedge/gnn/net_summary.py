"""Summary of the network responses on the charged borders (A58; regenerates charged_summary.json).

For each border, variant (graph, local) and period (charged = January-June 2026, y2025), volume-weighted means over
the covered export hours of the network-response model outputs (charged_<border>_<variant>.npz: rho = dR / delta and
r = signed emission response per delivered MWh removed, one row per trained model), reported as [mean over models,
min, max]:
  absorb_importer  dR_m / delta           share of a removed MWh that the importer's own plants supply
  absorb_exporter  -dR_x / delta          reduction of the exporter's own output
  E_x = -r_x, E_m = r_m, third = sum of r over third zones, third_EU = over third zones at the EU allowance price,
  network = sum of r over all zones
  third_by_zone    third zones with |r| >= 0.01 (mean over models)
  two_zone         accounting comparator from the same local responses m_i (mslope): E_x = m_x / eta, E_m = m_m,
                   network = E_m - E_x
Usage: WEDGE_GNN_SPEC=v4 NETRESP_TAG=r2 python src/wedge/gnn/net_summary.py
Output: <netresp OUTD>/charged_summary.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import SPLITS, Features, hour_of      # noqa: E402
from wedge.gnn.netresp import ETA, OUTD                        # noqa: E402

NON_EUA = ("GB", "RS", "BA", "ME", "MK")


def trip(v):
    return [round(float(v.mean()), 3), round(float(v.min()), 3), round(float(v.max()), 3)]


def main():
    fe = Features()
    idx = {n: i for i, n in enumerate(fe.nodes)}
    res = {}
    for b in ETA:
        x, m = b.split("->")
        xi, mi = idx[x], idx[m]
        for variant in ("graph", "local"):
            f = OUTD / f"charged_{b.replace('->', '_')}_{variant}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            hours, rho, r, ms, flow = z["hours"], z["rho"], z["r"], z["mslope"], z["flow"]
            for per in ("charged", "y2025"):
                a_, b_ = SPLITS[per]
                sel = (hours >= hour_of(a_)) & (hours < hour_of(b_))
                if sel.sum() == 0:
                    continue
                w = flow[sel] / flow[sel].sum()
                rs, rhos = r[:, sel].astype(np.float64), rho[:, sel].astype(np.float64)
                third = [j for j in range(fe.N) if j not in (xi, mi)]
                third_eu = [j for j in third if fe.nodes[j] not in NON_EUA]
                o = {"absorb_importer": trip(rhos[..., mi] @ w), "absorb_exporter": trip(-rhos[..., xi] @ w),
                     "E_x": trip(-rs[..., xi] @ w), "E_m": trip(rs[..., mi] @ w),
                     "third": trip(rs[..., third].sum(-1) @ w), "third_EU": trip(rs[..., third_eu].sum(-1) @ w),
                     "network": trip(rs.sum(-1) @ w), "hours": int(sel.sum())}
                bz = (rs[..., third].mean(0) * w[:, None]).sum(0)
                o["third_by_zone"] = {fe.nodes[third[k]]: round(float(v), 3) for k, v in enumerate(bz)
                                      if abs(v) >= 0.01}
                ex2, em2 = float(w @ ms[sel, xi]) / ETA[b], float(w @ ms[sel, mi])
                o["two_zone"] = {"E_x": round(ex2, 3), "E_m": round(em2, 3), "network": round(em2 - ex2, 3)}
                # exchange with zones outside the graph: total residual shift (1 - 1/eta) minus the summed output change
                o["boundary_outflow"] = trip(-((1 - 1 / ETA[b]) - rhos.sum(-1) @ w))
                res[f"{b.replace('->', '_')}|{variant}|{per}"] = o
                print(b, variant, per, o, flush=True)
    (OUTD / "charged_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
