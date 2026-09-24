"""Outage test of the Serbian network response on the RS-HU tie (A62).

Uses the estimator and model comparison of outage_multi.py on the link RS-HU (signed flow RS->HU), with outages that
start after June 2023. Primary: the outages whose zero flow JAO confirms as zero offered capacity in both directions
(spells starting 2024-09-09 and 2025-08-11). Sensitivity: all zero-flow outages after June 2023 (adds the spell
starting 2023-08-12, before JAO ran the HU-RS auctions).
Outputs: data/processed/gnn/outage_rs_v4.json (primary), outage_rs_all_v4.json (sensitivity), with block files.
Usage: WEDGE_GNN_SPEC=v4 NETRESP_TAG=r2 python src/wedge/gnn/outage_rs.py [primary|all]   (default: both)
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import hour_of          # noqa: E402
from wedge.gnn.netresp import ETA               # noqa: E402
from wedge.gnn import outage_multi as om        # noqa: E402

JAO_VERIFIED = ("2024-09-09", "2025-08-11")      # zero-flow spell starts with zero offered capacity on JAO


def verified(a, b, s, e):
    return any(abs(s - hour_of(v)) <= 48 for v in JAO_VERIFIED)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "primary"):
        om.main(links=[("RS", "HU")], held_start="2023-07-01", out_name="outage_rs", event_ok=verified,
                eta=ETA["RS->HU"], min_events=1)
    if which in ("both", "all"):
        om.main(links=[("RS", "HU")], held_start="2023-07-01", out_name="outage_rs_all", event_ok=None,
                eta=ETA["RS->HU"], min_events=1)
