"""Export- and import-onset curves behind the flow-onset comparison (A48; appendix figure).

Pools the four charged destination groups (NL, BE, FR links combined, DK1) and reports, per year (January-August),
the share of hours with exports (imports) above 5% of nominal capacity in 5 EUR/MWh bins of the realised spread,
keeping bins with at least 30 hours. Uses the same series and thresholds as did_d1.py.
Output: data/processed/onset_curves.json
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge.did_d1 import CONTROL, LINKS, onset, series                   # noqa: E402

BIN = 5.0


def main():
    out = {}
    for y in (2025, 2026):
        s = series(y)
        for side in ("exp", "imp"):
            b = defaultdict(list)
            for link in LINKS:
                if link == CONTROL:
                    continue
                for _, sp, f in onset(s[link], side):
                    b[round(sp / BIN) * BIN].append(f)
            out[f"{side}_{y}"] = {str(k): [sum(v) / len(v), len(v)] for k, v in sorted(b.items()) if len(v) >= 30}
    pathlib.Path("data/processed/onset_curves.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out.items():
        print(k, len(v), "bins")


if __name__ == "__main__":
    main()
