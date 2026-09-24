"""Bootstrap stability for the trading comparisons (A48, revision check concern 3).

1. Pooled flow-onset DDD with 1,000 joint week-block resamples (did_d1.pooled; the frozen 300-draw artefact
   did_d1_pooled.json is restored afterwards, the new result goes to did_d1_pooled_rev.json).
2. Capacity regression, main specification (cap_event_rev spec D: lag D-2, January-August of both years,
   link x direction x month-of-year effects), mean valuation and P(A > 0), 1,000 week-block resamples.
Output: data/processed/boot_stability.json
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import did_d1                                              # noqa: E402
from wedge.cap_event_rev import boot, fit, panel                      # noqa: E402

N = 1000


def main():
    out = {}
    frozen = pathlib.Path("data/processed/did_d1_pooled.json")
    keep = frozen.with_suffix(".json.keep")
    shutil.copyfile(frozen, keep)
    try:
        r = did_d1.pooled(n_boot=N, seed=13)
        out["flow_pooled_DDD"] = dict(point=r["pooled_DDD"], ci95=r["ci95"], reps=r["n_boot"])
        pathlib.Path("data/processed/did_d1_pooled_rev.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    finally:
        shutil.copyfile(keep, frozen)
        keep.unlink()
    print("flow", out["flow_pooled_DDD"], flush=True)
    rows = [r for r in panel(2, range(1, 9)) if not (r["year"] == 2026 and r["moy"] > 8)]
    for name, rr in (("capacity_mean", rows), ("capacity_P(A>0)", [dict(r, y=1.0 if r["y"] > 0 else 0.0) for r in rows])):
        b = fit(rr, True)
        bs = boot(rr, True, N, seed=101)
        out[name] = dict(point=b, ci95=(bs[int(0.025 * N)], bs[int(0.975 * N) - 1]), reps=N)
        print(name, out[name], flush=True)
    pathlib.Path("data/processed/boot_stability.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
