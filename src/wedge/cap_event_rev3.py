"""Per-link capacity coefficients under the revised specification (A48; Table 6).

Spec D of cap_event_rev (lag D-2, January-August of 2025 and 2026, link x direction x month-of-year effects),
estimated link by link, 200 week-block resamples.
Output: data/processed/cap_event_rev_links.json
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge.cap_event_rev import boot, fit, panel                      # noqa: E402


def main(n=200):
    rows = [r for r in panel(2, range(1, 9)) if not (r["year"] == 2026 and r["moy"] > 8)]
    out = {}
    for link in sorted({r["link"] for r in rows}):
        rr = [r for r in rows if r["link"] == link]
        b = fit(rr, True)
        bs = boot(rr, True, n, seed=43)
        out[link] = dict(beta=b, ci95=(bs[int(0.025 * n)], bs[int(0.975 * n) - 1]), reps=n)
        print(f"{link} {b:+.3f} [{out[link]['ci95'][0]:+.3f}, {out[link]['ci95'][1]:+.3f}]", flush=True)
    pathlib.Path("data/processed/cap_event_rev_links.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
