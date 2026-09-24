"""D-019 addendum B: event-study regression of JAO capacity prices.

A_{b,d,h} = f_{b,d}(lagged-spread bin) + lambda_{b,week} + psi_{b,d,hour-of-day}
            + sum_m beta_m 1{d = GB->EU} 1{month(h) = m} + e          (event study, ref Dec 2025)
          or + beta_post 1{d = GB->EU} 1{h in 2026}                     (pooled post)

Links NLL, IF1, IF2, EL1, VKL; both directions; Jan 2025 - Aug 2026; zeros kept. Lagged spread = the
same hour's realised spread on D-1 (known before the capacity auction), sign-adjusted per direction.
Sparse least squares (scipy lsqr). Inference: week-block bootstrap of the whole panel.
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


def panel():
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
                if not ((t.year == 2025) or (t.year == 2026 and t.month <= 8)):
                    continue
                hl = h - 86400
                if hl not in gbp or hl not in mp:
                    continue
                s = mp[hl] - gbp[hl]
                s = s if d == "exp" else -s
                rows.append(dict(link=prefix, d=d, h=h, y=a, lag=s,
                                 month=(t.year - 2025) * 12 + t.month,   # 1..20
                                 week=h // (7 * 86400), hod=t.hour))
    return rows


def design(rows, event=True):
    cols = {}

    def col(key):
        if key not in cols:
            cols[key] = len(cols)
        return cols[key]
    I, J = [], []
    treat = {}
    for i, r in enumerate(rows):
        b = int(min(max(r["lag"], -100), 150) // 5)
        for key in (("f", r["link"], r["d"], b), ("lam", r["link"], r["week"]),
                    ("psi", r["link"], r["d"], r["hod"])):
            I.append(i)
            J.append(col(key))
        if r["d"] == "exp":
            if event:
                if r["month"] != 12:                                     # ref: Dec 2025
                    k = ("beta", r["month"])
                    I.append(i)
                    J.append(col(k))
                    treat[k] = cols[k]
            elif r["month"] >= 13:
                I.append(i)
                J.append(col(("post",)))
                treat[("post",)] = cols[("post",)]
    X = csr_matrix((np.ones(len(I)), (I, J)), shape=(len(rows), len(cols)))
    y = np.array([r["y"] for r in rows])
    return X, y, treat


def fit(rows, event):
    X, y, treat = design(rows, event)
    sol = lsqr(X, y, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    return {k: float(sol[j]) for k, j in treat.items()}


def boot(rows, event, n=100, seed=23):
    wk = defaultdict(list)
    for r in rows:
        wk[r["week"]].append(r)
    keys = sorted(wk)
    rnd = random.Random(seed)
    out = defaultdict(list)
    for _ in range(n):
        smp = []
        for j, w in enumerate(rnd.choice(keys) for _ in keys):
            for r in wk[w]:
                smp.append(dict(r, week=(r["week"], j)))                 # resampled week = new FE
        for k, v in fit(smp, event).items():
            out[k].append(v)
    return {k: (sorted(v)[int(.025 * len(v))], sorted(v)[int(.975 * len(v))]) for k, v in out.items()}


def main():
    rows = panel()
    print(f"panel rows {len(rows):,}")
    res = {}
    post = fit(rows, False)[("post",)]
    post_ci = boot(rows, False, n=100)[("post",)]
    res["pooled_post"] = dict(beta=post, ci95=post_ci)
    print(f"POOLED beta_post (GB->EU vs EU->GB, 2026 vs 2025): {post:+.2f} EUR/MWh 95% CI "
          f"[{post_ci[0]:+.2f}, {post_ci[1]:+.2f}]")
    for L in LINKS:
        sub = [r for r in rows if r["link"] == L]
        b = fit(sub, False)[("post",)]
        ci = boot(sub, False, n=60)[("post",)]
        res[f"post_{L}"] = dict(beta=b, ci95=ci)
        print(f"   {L}: {b:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    ev = fit(rows, True)
    ev_ci = boot(rows, True, n=100)
    res["event"] = {str(k[1]): dict(beta=v, ci95=ev_ci.get(k)) for k, v in sorted(ev.items())}
    print("event-time beta_m (month 1 = Jan 2025, ref 12 = Dec 2025):")
    for m, v in res["event"].items():
        print(f"   m{int(m):2d} {v['beta']:+6.2f} [{v['ci95'][0]:+6.2f}, {v['ci95'][1]:+6.2f}]")
    pathlib.Path("data/processed/cap_event.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()


def distribution(thresholds=(0.0, 1.0, 2.0, 5.0, 10.0, 20.0), n_boot=60):
    """A43 (ruling #2 item 4): distribution regression. For each threshold c, the same design with
    outcome 1{A > c}: beta_post = change in P(A > c) for GB->EU relative to EU->GB, 2026 vs 2025."""
    rows = panel()
    out = {}
    for c in thresholds:
        r2 = [dict(r, y=1.0 if r["y"] > c else 0.0) for r in rows]
        b = fit(r2, False)[("post",)]
        ci = boot(r2, False, n=n_boot)[("post",)]
        out[str(c)] = dict(beta=b, ci95=ci)
        print(f"P(price > {c:5.1f}): outward-vs-inward change 2026 vs 2025 {b:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    pathlib.Path("data/processed/cap_distribution.json").write_text(json.dumps(out, indent=1),
                                                                    encoding="utf-8")
    return out
