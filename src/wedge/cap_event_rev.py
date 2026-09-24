"""Revision of the capacity-price regression (A48).

Changes against cap_event.py (which stays frozen as the registered/earlier specification):
  * lag      the GB price is Elexon's Market Index Price (APXMIDP, short-term trades), not a day-ahead auction price.
             The same-hour spread of D-1 is not fully known when the D-1 morning capacity auction closes, so the
             pre-auction control uses the same hour of D-2 (all D-2 hours are settled before the auction).
  * months   2025 and 2026 restricted to January-August, so both years have the same seasonal composition.
  * season   optional link x direction x month-of-year effects (direction-specific seasonality).
  * reps     week-block bootstrap with more resamples.
Outcomes: the pooled outward-vs-inward coefficient beta_post and P(A > 0) (distribution version).
Output: data/processed/cap_event_rev.json
"""
from __future__ import annotations

import json
import pathlib
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import run_year_gbfr as RG                                  # noqa: E402
from wedge.cap_price import LINKS, load_prices                         # noqa: E402
from wedge.run_year import _merge                                      # noqa: E402


def panel(lag_days=2, months=range(1, 9)):
    gbp, mps = {}, {}
    for y in (2025, 2026):
        _, p, *_ = RG.load_gb_fr(pathlib.Path(f"data/raw/gb_fr_{y}"), "nl")
        gbp.update(p)
    rows = []
    for prefix, (eu, pre, kind) in LINKS.items():
        mp = {}
        for y in (2025, 2026):
            mp.update(_merge(f"data/raw/{kind}_{y}/{pre}_price_*.json", "price")["price"])
        for d in ("exp", "imp"):
            for h, (a, off, alloc) in load_prices(prefix, eu, d).items():
                t = datetime.fromtimestamp(h, tz=timezone.utc)
                if t.year not in (2025, 2026) or t.month not in months:
                    continue
                hl = h - lag_days * 86400
                if hl not in gbp or hl not in mp:
                    continue
                s = mp[hl] - gbp[hl]
                s = s if d == "exp" else -s
                rows.append(dict(link=prefix, d=d, h=h, y=a, lag=s, year=t.year, moy=t.month,
                                 week=h // (7 * 86400), hod=t.hour))
    return rows


def design(rows, season):
    cols = {}

    def col(key):
        if key not in cols:
            cols[key] = len(cols)
        return cols[key]
    I, J = [], []
    for i, r in enumerate(rows):
        b = int(min(max(r["lag"], -100), 150) // 5)
        keys = [("f", r["link"], r["d"], b), ("lam", r["link"], r["week"]), ("psi", r["link"], r["d"], r["hod"])]
        if season:
            keys.append(("moy", r["link"], r["d"], r["moy"]))
        if r["d"] == "exp" and r["year"] == 2026:
            keys.append(("post",))
        for k in keys:
            I.append(i)
            J.append(col(k))
    X = csr_matrix((np.ones(len(I)), (I, J)), shape=(len(rows), len(cols)))
    return X, np.array([r["y"] for r in rows]), cols[("post",)]


def fit(rows, season):
    X, y, j = design(rows, season)
    return float(lsqr(X, y, atol=1e-10, btol=1e-10, iter_lim=20000)[0][j])


def boot(rows, season, n, seed=29):
    wk = defaultdict(list)
    for r in rows:
        wk[r["week"]].append(r)
    keys = sorted(wk)
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        smp = []
        for j, w in enumerate(rnd.choice(keys) for _ in keys):
            for r in wk[w]:
                smp.append(dict(r, week=(r["week"], j)))
        out.append(fit(smp, season))
    out.sort()
    return out


def main(n=400):
    res = {}
    specs = {
        "A draft (lag D-1, Jan25-Aug26, no season)": dict(lag=1, months=range(1, 13), season=False),
        "B lag D-2, Jan25-Aug26, no season": dict(lag=2, months=range(1, 13), season=False),
        "C lag D-2, Jan-Aug both years, no season": dict(lag=2, months=range(1, 9), season=False),
        "D lag D-2, Jan-Aug both years, link x dir x month-of-year": dict(lag=2, months=range(1, 9), season=True),
    }
    for name, sp in specs.items():
        rows = panel(sp["lag"], sp["months"])
        rows2026 = [r for r in rows if not (r["year"] == 2026 and r["moy"] > 8)]
        for outcome in ("A", "P(A>0)"):
            rr = rows2026 if outcome == "A" else [dict(r, y=1.0 if r["y"] > 0 else 0.0) for r in rows2026]
            b = fit(rr, sp["season"])
            bs = boot(rr, sp["season"], n)
            ci = (bs[int(0.025 * n)], bs[int(0.975 * n) - 1])
            half = boot(rr, sp["season"], n // 2, seed=31)
            ci_half = (half[int(0.025 * len(half))], half[int(0.975 * len(half)) - 1])
            res.setdefault(name, {})[outcome] = dict(beta=b, ci95=ci, ci95_other_seed_half_reps=ci_half,
                                                     n_rows=len(rr), reps=n)
            print(f"{name:62s} {outcome:7s} beta {b:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  (half-reps other seed "
                  f"[{ci_half[0]:+.3f}, {ci_half[1]:+.3f}])  n={len(rr)}", flush=True)
    pathlib.Path("data/processed/cap_event_rev.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
