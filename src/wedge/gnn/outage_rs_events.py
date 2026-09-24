"""Event-by-event Serbia-Hungary outage comparison (A64).

With two confirmed outages the event bootstrap has only three distinct resamples, so the Serbian comparison is reported
event by event. For each outage of outage_rs (primary: the two JAO-confirmed outages; outage_rs_all adds the
unconfirmed outage of August 2023) and each outcome (R and other-link net imports of RS and HU, dispatchable
emissions of RS and HU), this script computes, specification A, pooled dose:
  outage slope      within-block slope of the residual on the dose over the event's treated hours
  window slope      the same slope over the normal-operation windows of the event's blocks
  difference        outage slope minus window slope
  model slopes      graph network, node-local and two-zone baseline, same within-block regression of the model effect
                    on the dose over the event's treated hours (zero in the windows)
Inputs: data/processed/gnn/outage_rs{_all}_blocks_RS_HU_v4.npz and outage_rs{_all}_model_RS_HU_v4.npz (outage_multi.py
A64 additions: window residuals and per-hour model effects).
Output: data/processed/gnn/outage_rs_events_v4.json
Usage: WEDGE_GNN_SPEC=v4 python src/wedge/gnn/outage_rs_events.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wedge.gnn.features import DSFX                                   # noqa: E402
from wedge.gnn.outage_v2 import T0, block_stats, slopes                # noqa: E402

G = pathlib.Path("data/processed/gnn")
KEYS = ("R_a", "R_b", "NIother_a", "NIother_b", "E_a", "E_b")
MODELS = ("graph", "local", "one_for_one")


def pooled_slope(L, y, blk):
    """Within-block (specification A) pooled slope of y on L."""
    if len(L) < 3:
        return float("nan")
    _, bi = np.unique(blk, return_inverse=True)
    S = block_stats(L, y, bi, bi, bi.max() + 1).sum(0)
    return float(slopes(S)[2])


def main():
    out = {}
    for run in ("outage_rs", "outage_rs_all"):
        z = np.load(G / f"{run}_blocks_RS_HU{DSFX}.npz")
        mz = np.load(G / f"{run}_model_RS_HU{DSFX}.npz")
        assert np.array_equal(mz["hours"], z["hours"])
        frozen = json.loads((G / f"{run}{DSFX}.json").read_text(encoding="utf-8"))["RS-HU"]
        hours, blk, ev, L = z["hours"], z["block"], z["event"], z["L"]
        pblk, ppar, pL = z["p_block"], z["p_parent"], z["p_L"]
        res = {}
        for e in np.unique(ev):
            sel = ev == e
            blocks_e = np.unique(blk[sel])
            psel = np.isin(ppar, blocks_e)
            start = datetime.fromtimestamp(T0 + 3600 * int(hours[sel].min()), tz=timezone.utc).strftime("%Y-%m-%d")
            row = {"first_treated_day": start, "treated_hours": int(sel.sum()), "blocks": int(len(blocks_e)),
                   "window_hours": int(psel.sum()), "mean_dose_MW": float(L[sel].mean()), "outcomes": {}}
            for k in KEYS:
                ok, pok = z[f"ok_{k}"].astype(bool) & sel, z[f"pok_{k}"].astype(bool) & psel
                b_out = pooled_slope(L[ok], z[f"dY_{k}"][ok], blk[ok])
                b_win = pooled_slope(pL[pok], z[f"pdY_{k}"][pok], pblk[pok])
                mods = {m: pooled_slope(L[ok], mz[f"{m}__{k}"][ok], blk[ok]) for m in MODELS}
                row["outcomes"][k] = {"outage": b_out, "window": b_win, "difference": b_out - b_win, "model": mods}
            res[str(int(e))] = row
        # check: all events together reproduce the frozen pooled estimate (specification A)
        checks = {}
        for k in KEYS:
            ok, pok = z[f"ok_{k}"].astype(bool), z[f"pok_{k}"].astype(bool)
            d = pooled_slope(L[ok], z[f"dY_{k}"][ok], blk[ok]) - pooled_slope(pL[pok], z[f"pdY_{k}"][pok], pblk[pok])
            checks[k] = {"recomputed": d, "frozen": frozen["outcomes"][k]["A_block"]["did"][2]}
            assert abs(d - checks[k]["frozen"]) < 1e-9, (run, k, d, checks[k]["frozen"])
        out[run] = {"events": res, "check_all_events": checks}
        for e, row in res.items():
            print(run, e, row["first_treated_day"], row["treated_hours"], "h",
                  {k: (round(v["difference"], 2), round(v["model"]["graph"], 2), round(v["model"]["one_for_one"], 2))
                   for k, v in row["outcomes"].items()}, flush=True)
    (G / f"outage_rs_events{DSFX}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
