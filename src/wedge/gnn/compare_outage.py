"""Compare model-implied outage responses with the difference-in-slopes estimates (A52).

For each validated model round (netresp, netresp_r2, ...) and variant (graph, local, one_for_one), report the error
of the per-zone responses of dispatchable output (R) and emissions (E) against outage_did.json, over the zones with
estimates, plus the share of zones whose model value lies inside the DiD 95% bootstrap interval.
Output: data/processed/gnn/compare_outage.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

P = pathlib.Path("data/processed/gnn")


def main(rounds, late=False):
    did = json.loads((P / ("outage_did_late.json" if late else "outage_did.json")).read_text(encoding="utf-8"))
    out = {}
    for rnd in rounds:
        f = P / rnd / ("validate_outages_late.json" if late else "validate_outages.json")
        if not f.exists():
            continue
        v = json.loads(f.read_text(encoding="utf-8"))
        for b, vb in v.items():
            x, m = b.split("->")
            D = did[b]
            # empirical vectors: own-zone responses and emissions by zone
            emp_R = {x: D["R_x"], m: D["R_m"]}
            for n in ("DE", "NO", "FR"):
                if f"R_{n}" in D:
                    emp_R[n] = D[f"R_{n}"]
            emp_E = {k[2:]: D[k] for k in D if k.startswith("E_")}
            for name in ("graph", "local", "one_for_one"):
                if name not in vb:
                    continue
                R, E = vb[name]["R"], vb[name]["E"]
                res = {}
                for q, emp, pr in (("R", emp_R, R), ("E", emp_E, E)):
                    err = [pr.get(n, 0.0) - e["did"] for n, e in emp.items()]
                    inside = [e["did_ci95"][0] <= pr.get(n, 0.0) <= e["did_ci95"][1] for n, e in emp.items()]
                    res[f"rmse_{q}"] = float(np.sqrt(np.mean(np.square(err))))
                    res[f"inside95_{q}"] = float(np.mean(inside))
                res["E_network_model"] = float(sum(E.values()))
                res["E_network_did"] = float(sum(e["did"] for e in emp_E.values()))
                res["importer_own_R"] = (R.get(m, 0.0), D["R_m"]["did"])
                res["exporter_own_R"] = (R.get(x, 0.0), D["R_x"]["did"])
                out[f"{rnd}|{b}|{name}"] = res
                print(f"{rnd:12s} {b} {name:12s} R rmse {res['rmse_R']:.3f} in95 {res['inside95_R']:.2f} | "
                      f"E rmse {res['rmse_E']:.3f} in95 {res['inside95_E']:.2f} | Enet model {res['E_network_model']:.2f}"
                      f" did {res['E_network_did']:.2f} | own m {res['importer_own_R'][0]:.2f}/{res['importer_own_R'][1]:.2f}"
                      f" x {res['exporter_own_R'][0]:.2f}/{res['exporter_own_R'][1]:.2f}", flush=True)
    (P / ("compare_outage_late.json" if late else "compare_outage.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if x != "--late"]
    main(args or ["netresp", "netresp_r2"], late="--late" in sys.argv)
