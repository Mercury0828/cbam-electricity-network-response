"""Does the CBAM charge actually move electricity flows? A reduced-form test using the policy's timing.

The charge applies to 2026 imports, not to 2025 (A25). If traders price it in, a trader imports
only when the spread (importer price - exporter price) exceeds the charge, so the curve
    F(s) = mean(export flow / capacity | spread s)
should shift RIGHT by roughly the effective charge between 2025 and 2026. Both years use the SAME
months (Jan-Aug) to hold seasonality fixed.

Estimand: Delta, the horizontal shift that best aligns F_2026(s) with F_2025(s - Delta), fitted by
least squares over common spread bins. Interpretation: the effective, priced-in per-MWh charge.
Compare with the gross charge (GB ~ EUR 32/MWh, RS ~ EUR 78/MWh in 2026 Q1-Q2) and with zero.
Week-block bootstrap (168 h, D-009) for the uncertainty.

🔴 This is reduced form. Other 2025 -> 2026 changes (outages, capacity, ETS prices on both sides)
can shift F too; conditioning on the spread absorbs price-driven changes but not capacity changes.
It identifies whether flows respond to the charge at the border level, not a structural elasticity.
"""
from __future__ import annotations

import json
import pathlib
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import run_year as RY, run_year_gbfr as RG     # noqa: E402
from wedge.regime import CAP                              # noqa: E402

BIN = 5.0
GRID = [x * 1.0 for x in range(-30, 101)]     # candidate shifts, EUR/MWh


def series(year, border):
    if border.startswith("GB"):
        imp = border.split("->")[1].lower()
        xg, xp, mg, mp, flow, *_ = RG.load_gb_fr(pathlib.Path(f"data/raw/gb_fr_{year}"), imp)
        cap = CAP[imp]
        exp = {h: -v for h, v in flow.items()}           # + = GB -> importer
    else:
        rs, hu, xp, mp, flow, *_ = RY.load_rs_hu(pathlib.Path(f"data/raw/rs_hu_{year}"))
        cap = 800.0                                        # max 2025 monthly NTC (EMS), declared
        exp = flow
    out = []
    for h in set(xp) & set(mp) & set(exp):
        d = datetime.fromtimestamp(h, tz=timezone.utc)
        if d.year != year or d.month > 8:
            continue
        out.append((h, mp[h] - xp[h], max(min(exp[h] / cap, 1.0), -1.0)))
    return out


def curve(obs):
    b = defaultdict(list)
    for _, s, f in obs:
        b[round(s / BIN) * BIN].append(f)
    return {k: (sum(v) / len(v), len(v)) for k, v in b.items() if len(v) >= 10}


def fit_shift(c25, c26):
    """Least-squares shift, weighted by bin counts, over bins present in both after shifting."""
    best = None
    for d in GRID:
        sse = w = 0.0
        for s, (f26, n26) in c26.items():
            k = round((s - d) / BIN) * BIN
            if k in c25:
                f25, n25 = c25[k]
                wt = min(n26, n25)
                sse += wt * (f26 - f25) ** 2
                w += wt
        if w > 200 and (best is None or sse / w < best[1]):
            best = (d, sse / w)
    return None if best is None else best[0]


def boot(o25, o26, n=200, seed=1):
    def weeks(o):
        g = defaultdict(list)
        for r in o:
            g[r[0] // (168 * 3600)].append(r)
        return list(g.values())
    w25, w26 = weeks(o25), weeks(o26)
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        s25 = [r for _ in w25 for r in rnd.choice(w25)]
        s26 = [r for _ in w26 for r in rnd.choice(w26)]
        d = fit_shift(curve(s25), curve(s26))
        if d is not None:
            out.append(d)
    out.sort()
    return out[int(.025 * len(out))], out[int(.975 * len(out))]


def band_diff(c25, c26, lo, hi):
    """Mean flow-share difference 2026 - 2025 in spread band [lo, hi)."""
    xs = [(c26[k][0] - c25[k][0], min(c26[k][1], c25[k][1])) for k in c26
          if lo <= k < hi and k in c25]
    w = sum(n for _, n in xs)
    return None if not w else sum(d * n for d, n in xs) / w


def main():
    res = {}
    for border, gross in (("GB->NL", 32.4), ("GB->BE", 32.4), ("GB->FR", 32.4), ("RS->HU", 78.4)):
        o25, o26 = series(2025, border), series(2026, border)
        c25, c26 = curve(o25), curve(o26)
        d = fit_shift(c25, c26)
        lo, hi = boot(o25, o26)
        pivotal = band_diff(c25, c26, 0.0, gross)
        control = band_diff(c25, c26, gross + 10, gross + 60)
        res[border] = dict(gross_charge=gross, shift_eur=d, shift_ci95=[lo, hi],
                           flow_share_diff_in_0_to_c=pivotal,
                           flow_share_diff_control_band=control, n2025=len(o25), n2026=len(o26))
        print(f"{border}: fitted shift {d} EUR/MWh [95% {lo}, {hi}] vs gross charge {gross} | "
              f"flow-share diff 2026-2025: spread in [0,c) {pivotal if pivotal is None else round(pivotal,3)}, "
              f"control band {control if control is None else round(control,3)} | n {len(o25)}/{len(o26)}")
    pathlib.Path("data/processed/spread_shift.json").write_text(json.dumps(res, indent=1),
                                                                encoding="utf-8")


if __name__ == "__main__":
    main()
