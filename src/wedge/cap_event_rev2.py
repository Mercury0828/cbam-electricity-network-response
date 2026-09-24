"""Fig. 7 under the revised capacity specification (A48).

(a) distribution version: outcome 1{A > c}, c in {0, 1, 2, 5, 10, 20}, spec D of cap_event_rev (lag D-2, January-August
    of 2025 and 2026, link x direction x month-of-year effects).
(b) event study: lag D-2, January 2025 - August 2026, one outward-direction term per month, reference December 2025,
    no month-of-year effects (they are collinear with the monthly terms in the first year).
Output: data/processed/cap_distribution_rev.json, data/processed/cap_event_study_rev.json
"""
from __future__ import annotations

import json
import pathlib
import random
import sys
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge.cap_event_rev import boot, fit, panel                      # noqa: E402


def event_design(rows):
    cols = {}

    def col(key):
        if key not in cols:
            cols[key] = len(cols)
        return cols[key]
    I, J, treat = [], [], {}
    for i, r in enumerate(rows):
        b = int(min(max(r["lag"], -100), 150) // 5)
        keys = [("f", r["link"], r["d"], b), ("lam", r["link"], r["week"]), ("psi", r["link"], r["d"], r["hod"])]
        m = (r["year"] - 2025) * 12 + r["moy"]
        if r["d"] == "exp" and m != 12:
            keys.append(("beta", m))
        for k in keys:
            I.append(i)
            J.append(col(k))
            if k[0] == "beta":
                treat[k] = cols[k]
    X = csr_matrix((np.ones(len(I)), (I, J)), shape=(len(rows), len(cols)))
    return X, np.array([r["y"] for r in rows]), treat


def event_fit(rows):
    X, y, treat = event_design(rows)
    sol = lsqr(X, y, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    return {k[1]: float(sol[j]) for k, j in treat.items()}


def event_boot(rows, n, seed=37):
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
                smp.append(dict(r, week=(r["week"], j)))
        for m, v in event_fit(smp).items():
            out[m].append(v)
    return {m: (sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1]) for m, v in out.items()}


def main(n_dist=200, n_event=150):
    rows = [r for r in panel(2, range(1, 9)) if not (r["year"] == 2026 and r["moy"] > 8)]
    dist = {}
    for c in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0):
        rr = [dict(r, y=1.0 if r["y"] > c else 0.0) for r in rows]
        b = fit(rr, True)
        bs = boot(rr, True, n_dist, seed=41)
        dist[str(c)] = dict(beta=b, ci95=(bs[int(0.025 * n_dist)], bs[int(0.975 * n_dist) - 1]), reps=n_dist)
        print(f"P(A>{c:4.1f}) {b:+.4f} [{dist[str(c)]['ci95'][0]:+.4f}, {dist[str(c)]['ci95'][1]:+.4f}]", flush=True)
    pathlib.Path("data/processed/cap_distribution_rev.json").write_text(json.dumps(dist, indent=1), encoding="utf-8")
    ev_rows = [r for r in panel(2, range(1, 13)) if not (r["year"] == 2026 and r["moy"] > 8)]
    pt = event_fit(ev_rows)
    ci = event_boot(ev_rows, n_event)
    ev = {str(m): dict(beta=pt[m], ci95=ci[m]) for m in sorted(pt)}
    pathlib.Path("data/processed/cap_event_study_rev.json").write_text(json.dumps(ev, indent=1), encoding="utf-8")
    print("event study done", flush=True)


if __name__ == "__main__":
    main()
