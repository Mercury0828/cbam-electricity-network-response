"""Regression check of the outage_v2 refactor (design / estimate split used by outage_synthetic.py).

Rebuilds the event blocks, pseudo windows and bootstrap draws with outage_v2.design, checks them against the frozen
block files, and recomputes every full-specification outcome from the frozen residuals and doses with
outage_v2.estimate; outage, pseudo and difference slopes, intervals and leave-one-event-out bounds must equal the frozen
outage_v2{DSFX}.json. Needs the frozen v4 artefacts (skipped otherwise).
Usage: WEDGE_GNN_SPEC=v4 python -m pytest tests/test_outage_v2_refactor.py -q
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("WEDGE_GNN_SPEC", "v4")
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from wedge.gnn.features import DSFX  # noqa: E402

FROZEN = ROOT / f"data/processed/gnn/outage_v2{DSFX}.json"
pytestmark = pytest.mark.skipif(not FROZEN.exists(), reason="frozen outage_v2 artefacts missing")


def test_design_and_estimate_reproduce_frozen_run():
    from wedge.gnn.features import Features
    from wedge.gnn.outage_v2 import design, drivers, estimate, outcome_ok
    fe = Features()
    d = np.load(f"data/processed/gnn/dataset{DSFX}.npz")
    idx = {n: i for i, n in enumerate(fe.nodes)}
    _, _, _, _, E, R, NI = drivers(fe, d)
    DS = design(fe, d, np.random.default_rng(11))
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    worst = 0.0
    for b, D in DS.items():
        z = np.load(f"data/processed/gnn/outage_v2_blocks_{b.replace('->', '_')}{DSFX}.npz")
        so, sp = D["so_all"], D["sp_all"]
        assert np.array_equal(np.where(so)[0], z["hours"]) and np.array_equal(D["blk"][so], z["block"])
        assert np.array_equal(np.where(sp)[0], z["p_hours"]) and np.array_equal(D["pid"][sp], z["p_block"])
        assert np.array_equal(D["par_of_pblock"][D["pid"][sp]], z["p_parent"])
        Luse, Fh = np.full(fe.H, np.nan), np.full(fe.H, np.nan)
        Luse[so], Fh[sp] = z["L"], z["p_L"]
        F = np.nan_to_num(D["F"])
        mi, xi = D["mi"], D["xi"]
        ys = {"R_m": R[mi], "R_x": R[xi], "NIother_m": NI[mi] - F, "NIother_x": NI[xi] + F}
        for n in fe.nodes:
            ys[f"E_{n}"], ys[f"R_{n}"] = E[idx[n]], R[idx[n]]
        for k, Y in ys.items():
            dY = np.zeros(fe.H)
            dY[so], dY[sp] = z[f"dY_{k}"], z[f"pdY_{k}"]
            out_k, _ = estimate(dY, outcome_ok(fe, idx, k, D, Y), D, Luse, Fh)
            ref = frozen[b]["outcomes"][f"{k}|full"]
            for fes in ("A_block", "B_block_hour"):
                for key in ("outage", "pseudo", "did", "ci95", "loo_min", "loo_max"):
                    a, r = np.array(out_k[fes][key], float), np.array(ref[fes][key], float)
                    assert np.array_equal(np.isnan(a), np.isnan(r)), (b, k, fes, key)
                    diff = np.nanmax(np.abs(a - r)) if np.isfinite(a).any() else 0.0
                    worst = max(worst, diff)
                    assert diff < 1e-9, (b, k, fes, key, diff)
    print("max abs deviation", worst)


if __name__ == "__main__":
    test_design_and_estimate_reproduce_frozen_run()
    print("PASS")
