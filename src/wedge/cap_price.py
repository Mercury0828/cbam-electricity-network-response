"""D-019 (pre-registered): is the CBAM charge capitalised into GB->EU explicit capacity prices? (A37)

JAO daily explicit auctions, hourly products, both directions: Nemo (NLL), IFA (IF1), IFA2 (IF2),
ElecLink (EL1), Viking (VKL). BritNed's daily auctions run on its own Empire platform, whose public
API retains only the current auction -> BritNed is not available (recorded, A37).

Hour mapping: blocks are numbered by LOCAL CLOCK hour (spring: B03 absent; autumn: B03 + B03DST);
sorted real-time order gives UTC = marketPeriodStart + position (verified on DST days, `check_dst`).

Outcome: marginal auctionPrice (EUR/MW per hour). Conditioning: realised spread s = p_EU - p_GB for
GB->EU, -s for EU->GB (A35 sources). Estimator: the A35 horizontal-shift estimator on the
price-spread curve (5 EUR bins), directional DDD = shift(GB->EU) - shift(EU->GB), week-block
bootstrap; pooled over links with a joint bootstrap. Plus descriptive price by spread band and the
congestion-revenue counterfactual.
"""
from __future__ import annotations

import glob
import json
import pathlib
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import run_year_gbfr as RG                                  # noqa: E402
from wedge.run_year import _merge                                      # noqa: E402
from wedge.spread_shift import curve, fit_shift                        # noqa: E402

LINKS = {  # prefix: (EU code in corridor, importer price prefix, raw kind)
    "NLL": ("BE", "be", "gb_fr"),
    "IF1": ("FR", "fr", "gb_fr"),
    "IF2": ("FR", "fr", "gb_fr"),
    "EL1": ("FR", "fr", "gb_fr"),
    "VKL": ("D1", "dk", "gb_ctrl"),
}
BLOCK = re.compile(r"B(\d{2})")


def load_prices(prefix, eu, direction):
    """{utc_hour: (price, offered, allocated)} for corridor prefix-direction."""
    corr = f"{prefix}-GB-{eu}" if direction == "exp" else f"{prefix}-{eu}-GB"
    out = {}
    for f in sorted(glob.glob(f"data/raw/jao/jao_Daily_{corr}_*.json")):
        for a in json.loads(pathlib.Path(f).read_text(encoding="utf-8")):
            if a.get("cancelled"):
                continue
            t0 = datetime.fromisoformat(a["marketPeriodStart"].replace("Z", "+00:00"))
            # Blocks are numbered by LOCAL CLOCK hour (spring day: B03 absent; autumn day: B03 and
            # B03DST). Sorting by (clock hour, DST repeat) gives real-time order, and the position
            # in that order is the elapsed hour from marketPeriodStart (verified on DST days).
            res = [r for r in a.get("results") or []
                   if BLOCK.match(r.get("productIdentification") or "")]
            res.sort(key=lambda r: (int(BLOCK.match(r["productIdentification"]).group(1)),
                                    "DST" in r["productIdentification"]))
            for k, r in enumerate(res):
                if r.get("auctionPrice") is None:
                    continue
                h = int((t0 + timedelta(hours=k)).timestamp())
                out[h] = (float(r["auctionPrice"]), r.get("offeredCapacity"),
                          r.get("allocatedCapacity"))
    return out


def check_dst():
    """Blocks per market day around the 2025/2026 DST switches (expect 23 and 25)."""
    for prefix, (eu, *_ ) in LINKS.items():
        for f in sorted(glob.glob(f"data/raw/jao/jao_Daily_{prefix}-GB-{eu}_*.json")):
            for a in json.loads(pathlib.Path(f).read_text(encoding="utf-8")):
                day = a["marketPeriodStart"][:10]
                if day in ("2025-03-29", "2025-10-25", "2026-03-28", "2026-10-24"):
                    blocks = sorted(r["productIdentification"][:3] for r in a.get("results") or [])
                    print(prefix, day, len(blocks), blocks[:1], blocks[-1:])


def series(year):
    _, gb_p, *_ = RG.load_gb_fr(pathlib.Path(f"data/raw/gb_fr_{year}"), "nl")
    out = {}
    for prefix, (eu, pre, kind) in LINKS.items():
        mp = _merge(f"data/raw/{kind}_{year}/{pre}_price_*.json", "price")["price"]
        for direction in ("exp", "imp"):
            cp = load_prices(prefix, eu, direction)
            obs = []
            for h, (p, off, alloc) in cp.items():
                d = datetime.fromtimestamp(h, tz=timezone.utc)
                if d.year != year or d.month > 8 or h not in gb_p or h not in mp:
                    continue
                s = mp[h] - gb_p[h]
                obs.append((h, s if direction == "exp" else -s, p, alloc or 0.0))
            out[(prefix, direction)] = obs
    return out


def shift_price(a, b):
    return fit_shift(curve([(h, s, p) for h, s, p, _ in a]), curve([(h, s, p) for h, s, p, _ in b]))


def ddd(s25, s26, prefix):
    e = shift_price(s25[(prefix, "exp")], s26[(prefix, "exp")])
    i = shift_price(s25[(prefix, "imp")], s26[(prefix, "imp")])
    return None if e is None or i is None else (e - i, e, i)


def bands(s25, s26):
    edges = [(-1e9, 0), (0, 5), (5, 15), (15, 32.4), (32.4, 60), (60, 1e9)]
    out = {}
    for prefix in LINKS:
        rows = []
        for lo, hi in edges:
            r = []
            for s in (s25, s26):
                v = [p for _, sp, p, _ in s[(prefix, "exp")] if lo <= sp < hi]
                r.append((len(v), sum(v) / len(v) if v else None))
            rows.append(dict(band=[lo, hi], n=[r[0][0], r[1][0]], mean_price=[r[0][1], r[1][1]]))
        out[prefix] = rows
    return out


def main(n_boot=300, seed=17):
    s25, s26 = series(2025), series(2026)
    res = {"links": {}}
    wk = lambda h: h // (168 * 3600)                                     # noqa: E731
    for prefix in LINKS:
        n = {k: (len(s25[(prefix, k)]), len(s26[(prefix, k)])) for k in ("exp", "imp")}
        d = ddd(s25, s26, prefix)
        res["links"][prefix] = dict(n=n, DDD=d and d[0], shift_exp=d and d[1], shift_imp=d and d[2])
        print(f"{prefix}: n exp {n['exp']} imp {n['imp']} | DDD {d}")
    # pooled, joint week-block bootstrap
    live = [p for p in LINKS if res["links"][p]["DDD"] is not None]
    point = sum(res["links"][p]["DDD"] for p in live) / len(live)
    g = {y: defaultdict(lambda: defaultdict(list)) for y in (2025, 2026)}
    for y, s in ((2025, s25), (2026, s26)):
        for k, obs in s.items():
            for r in obs:
                g[y][wk(r[0])][k].append(r)
    rnd = random.Random(seed)
    w25, w26 = sorted(g[2025]), sorted(g[2026])
    bs = []
    for _ in range(n_boot):
        b25 = [rnd.choice(w25) for _ in w25]
        b26 = [rnd.choice(w26) for _ in w26]
        a25 = defaultdict(list)
        a26 = defaultdict(list)
        for w in b25:
            for k, v in g[2025][w].items():
                a25[k].extend(v)
        for w in b26:
            for k, v in g[2026][w].items():
                a26[k].extend(v)
        v = [ddd(a25, a26, p) for p in live]
        if all(x is not None for x in v):
            bs.append(sum(x[0] for x in v) / len(v))
    bs.sort()
    ci = (bs[int(.025 * len(bs))], bs[int(.975 * len(bs))])
    res["pooled"] = dict(links=live, DDD=point, ci95=ci, n_boot=len(bs))
    res["P-D19a_capitalisation"] = "HOLDS" if ci[0] > 0 else "FALSIFIED"
    res["bands_exp_price"] = bands(s25, s26)
    print(f"POOLED capacity-price DDD over {live}: {point:.2f} EUR/MWh, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]"
          f" -> P-D19a {res['P-D19a_capitalisation']}")
    for p, rows in res["bands_exp_price"].items():
        print(p, " | ".join(f"[{r['band'][0]:.0f},{r['band'][1]:.0f}) "
                            f"{r['mean_price'][0] if r['mean_price'][0] is None else round(r['mean_price'][0], 1)}"
                            f"->{r['mean_price'][1] if r['mean_price'][1] is None else round(r['mean_price'][1], 1)}"
                            for r in rows))
    pathlib.Path("data/processed/cap_price.json").write_text(json.dumps(res, indent=1),
                                                             encoding="utf-8")


if __name__ == "__main__":
    if "--dst" in sys.argv:
        check_dst()
    else:
        main()
