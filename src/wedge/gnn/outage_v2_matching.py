"""Pseudo-window matching actually achieved by outage_v2 (A60).

Each outage block asks for five non-overlapping normal-operation windows of exactly its length that start in the same
calendar quarter, within one year where possible (otherwise any year). Long outages fill the quarter, so some blocks
receive fewer windows or none. This script records, per focal link, the number of blocks, the blocks with at least
one window, the windows found, the relaxed and the failed requests.
Output: data/processed/gnn/outage_v2_matching_v4.json
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

G = pathlib.Path("data/processed/gnn")


def main():
    v2 = json.loads((G / "outage_v2_v4.json").read_text(encoding="utf-8"))
    out = {}
    for b in ("GB->NL", "GB->BE"):
        z = np.load(G / f"outage_v2_blocks_{b.replace('->', '_')}_v4.npz")
        out[b] = {"blocks": int(z["block"].max()) + 1, "blocks_with_window": int(len(np.unique(z["p_parent"]))),
                  "windows": int(z["p_block"].max()) + 1, "requested": 5 * (int(z["block"].max()) + 1),
                  "relaxed": v2[b]["pseudo_relaxed"], "failed": v2[b]["pseudo_failed"],
                  "window_hours": int(len(z["p_hours"])), "treated_hours": int(len(z["hours"]))}
    (G / "outage_v2_matching_v4.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
