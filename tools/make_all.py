"""Regenerate every artefact the paper cites, in dependency order, from the raw data (self-check T1).

Usage:  python tools/make_all.py [--check]
--check: hash data/processed/*.json before and after the run and report any artefact whose bytes
changed (regeneration must be byte-identical: every bootstrap is seeded, every solver deterministic).
Raw downloads are NOT refetched here (see src/wedge/fetch/*; every download is logged with URL and
sha256 in data/manifest.jsonl); the processed fuel tables are verified against their primary files by
build_inputs.py first.
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
STEPS = [
    ["src/wedge/build_inputs.py"],                       # verify fuel tables vs primary files
    ["src/wedge/run_tiers.py"],
    ["src/wedge/run_tiers.py", "--independent"],
    ["src/wedge/regime.py"],
    ["src/wedge/tierB_mechanism.py"],
    ["src/wedge/attrition.py"],
    ["src/wedge/frontier.py"],
    ["src/wedge/d4_rules.py"],
    ["src/wedge/wedge.py"],
    ["src/wedge/benchmark.py"],
    ["src/wedge/audit_benchmark.py"],
    ["src/wedge/audit_benchmark.py", "--scale=0.457,0.423"],
    ["src/wedge/d2_frontier.py"],
    ["src/wedge/regime_map.py"],
    ["src/wedge/aggregate.py"],
    ["src/wedge/spread_shift.py"],
    ["-c", "import sys; sys.path.insert(0,'src'); from wedge import did_d1 as D; D.main(); D.directional(); D.pooled(); D.recovery()"],
    ["src/wedge/cap_price.py"],
    ["src/wedge/cap_event.py"],
    ["-c", "import sys; sys.path.insert(0,'src'); from wedge.cap_event import distribution; distribution()"],
    ["src/wedge/r2_cert.py"],
    ["src/wedge/val_envelope.py"],
    ["src/wedge/val_regress.py"],
    ["-c", "import sys; sys.path.insert(0,'src'); from wedge.val_regress import rs_hu_main, daily_main, model_rs_counterpart; rs_hu_main(); daily_main(); model_rs_counterpart()"],
    ["-c", "import sys; sys.path.insert(0,'src'); from wedge.val_regress import be_main; be_main()"],
    ["src/wedge/val_coal_reversal.py"],
    ["src/wedge/t2_ablation.py"],
    ["src/wedge/t4_baselines.py"],
    ["src/wedge/figures.py"],
    ["src/wedge/headline_numbers.py"],
]


def hashes():
    return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted((ROOT / "data/processed").glob("*.json"))}


def main():
    check = "--check" in sys.argv
    before = hashes() if check else {}
    for step in STEPS:
        t = time.time()
        r = subprocess.run([PY, *step], cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        print(f"[{'OK ' if r.returncode == 0 else 'ERR'}] {' '.join(step)[:90]}  ({time.time() - t:.0f}s)")
        if r.returncode != 0:
            print(r.stderr[-2000:])
            return 1
    if check:
        after = hashes()
        # R1-08: hashes() returns (sha256, mtime). Only files the run actually REWROTE (mtime moved)
        # count as regenerated; byte identity is judged on the sha256 alone.
        rewritten = [k for k in after if k in before and after[k][1] != before[k][1]]
        untouched = [k for k in after if k in before and after[k][1] == before[k][1]]
        changed = [k for k in rewritten if before[k][0] != after[k][0]]
        new = [k for k in after if k not in before]
        print(f"\nregenerated {len(rewritten)} artefacts: {len(rewritten) - len(changed)} byte-identical, "
              f"{len(changed)} CHANGED | {len(new)} new | {len(untouched)} not produced by this pipeline")
        for k in changed:
            print("  CHANGED:", k)
        for k in untouched:
            print("  not regenerated (legacy / not cited):", k)
        return 1 if changed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
