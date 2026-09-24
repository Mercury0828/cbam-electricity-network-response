"""Error summaries of the model-side outage responses by outcome group (A58).

From outage_model_compare_v4.json (rows per outcome: model slope, empirical difference in slopes, 95% interval), the
root-mean-square error of model minus outage estimate and the share of outcomes inside the interval, per link,
direction (plus, minus, pooled), specification and model, for three groups:
  output          dispatchable output of every zone with an estimate (R_<zone>; the endpoint aliases R_m and R_x are
                  the same series as R_<importer> and R_<exporter> and are not counted twice)
  output_trade    output plus the net imports of the two end zones on their other links (NIother_m, NIother_x)
  emissions       dispatchable emissions of every zone with an estimate (E_<zone>)
Output: data/processed/gnn/outage_model_summary_v4.json
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

G = pathlib.Path("data/processed/gnn")


def main():
    import os
    tag = os.environ.get("NETRESP_TAG", "r2")
    rtag = "" if tag in ("", "r2") else f"_{tag}"
    d = json.loads((G / f"outage_model_compare{rtag}_v4.json").read_text(encoding="utf-8"))
    out = {}
    for b, byname in d.items():
        for name, blob in byname.items():
            rows = blob["rows"]
            groups = {"output": [k for k in rows if k.startswith("R_") and k not in ("R_m", "R_x")],
                      "output_trade": [k for k in rows if (k.startswith("R_") and k not in ("R_m", "R_x"))
                                       or k.startswith("NIother")],
                      "emissions": [k for k in rows if k.startswith("E_")]}
            for fes in ("A_block", "B_block_hour"):
                for j, lab in enumerate(("plus", "minus", "pooled")):
                    for grp, keys in groups.items():
                        errs, ins = [], []
                        for k in keys:
                            e = rows[k][fes]
                            m, v = e["model"][j], e["empirical_did"][j]
                            if m is None or v is None or not (np.isfinite(m) and np.isfinite(v)):
                                continue
                            errs.append(m - v)
                            if e["inside"][j] is not None:
                                ins.append(e["inside"][j])
                        out[f"{b}|{name}|{fes}|{lab}|{grp}"] = {
                            "rmse": float(np.sqrt(np.mean(np.square(errs)))) if errs else None,
                            "inside_share": float(np.mean(ins)) if ins else None, "n": len(errs)}
    (G / f"outage_model_summary{rtag}_v4.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for b in d:
        for lab in ("plus", "pooled"):
            print(b, lab, {grp: [round(out[f"{b}|{n}|A_block|{lab}|{grp}"]["rmse"], 3) for n in d[b]]
                           for grp in ("output", "output_trade", "emissions")})


if __name__ == "__main__":
    main()
