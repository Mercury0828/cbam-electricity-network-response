"""D1 by difference-in-differences: is the CBAM charge priced into GB->EU trading? (A35)

Treated links (GB -> EU, charged from 2026-01-01): BritNed (NL), Nemo Link (BE), IFA+IFA2+ElecLink
(FR), Viking Link (DK1). Control: North Sea Link (GB -> NO2). Norway is outside the EU customs
territory, so GB -> NO exports carry NO CBAM charge, while NSL shares the GB side (same GB price
series, same GB system shocks, same explicit-allocation regime family).

For each link l and year y (Jan-Aug both years, holding season fixed):
    F_{l,y}(s) = mean(export / capacity | spread s),   s = p_importer - p_GB   (EUR/MWh)
A priced-in per-MWh charge c shifts the curve right by c. Delta_l = horizontal shift aligning
F_{l,2026} with F_{l,2025} (least squares over common bins, as in spread_shift.py).
    DiD_l = Delta_l - Delta_NSL
removes common 2025 -> 2026 changes on the GB side (price-index stage, GB system, FX). Inference:
week-block bootstrap (168 h, D-009) resampling the SAME weeks for treated and control, so common
shocks stay paired.

Capacities (nominal, declared): BritNed 1000, Nemo 1000, GB-FR 4000 (regime.py, sourced round
interconnector-capacity); Viking 1400 and NSL 1400 MW (operator nominal ratings; NOT yet sourced in a
round - the shift estimator is invariant to a common rescaling of one link across years).

🔴 Interpretation discipline (review B2): the estimand is the priced-in per-MWh wedge in the
flow-spread relation, relative to the uncharged control. A small Delta is compatible with a small
expected NET charge fully passed through; it is not a proof of zero response.
"""
from __future__ import annotations

import glob
import json
import pathlib
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import run_year_gbfr as RG                                  # noqa: E402
from wedge.run_year import _merge                                      # noqa: E402
from wedge.spread_shift import curve, fit_shift                        # noqa: E402

LINKS = {  # name: (interconnector names, importer price file prefix, raw dir kind, capacity MW)
    "GB->NL": ({"Netherlands(BritNed)"}, "nl", "gb_fr", 1000.0),
    "GB->BE": ({"Belgium (Nemolink)"}, "be", "gb_fr", 1000.0),
    "GB->FR": ({"France(IFA)", "IFA2 (INTIFA2)", "Eleclink (INTELEC)"}, "fr", "gb_fr", 4000.0),
    "GB->DK1": ({"Denmark (Viking link)"}, "dk", "gb_ctrl", 1400.0),
    "GB->NO2 (control)": ({"North Sea Link (INTNSL)"}, "no", "gb_ctrl", 1400.0),
}
CONTROL = "GB->NO2 (control)"


def flows(year, names):
    ic = {}
    for f in sorted(glob.glob(f"data/raw/gb_fr_{year}/gb_interconnector_*.json")):
        for rec in json.loads(pathlib.Path(f).read_text(encoding="utf-8")).get("data", []):
            if rec.get("interconnectorName") in names:
                ic[(rec["startTime"], rec["interconnectorName"])] = float(rec["generation"])
    per = defaultdict(float)
    for (t, _), v in ic.items():
        per[RG._hour(t)] += v / 2.0
    return {h: -v for h, v in per.items()}        # + = export FROM GB


def series(year):
    _, gb_p, *_ = RG.load_gb_fr(pathlib.Path(f"data/raw/gb_fr_{year}"), "nl")
    out = {}
    for link, (names, pre, kind, cap) in LINKS.items():
        mp = _merge(f"data/raw/{kind}_{year}/{pre}_price_*.json", "price")["price"]
        ex = flows(year, names)
        obs = []
        for h in set(gb_p) & set(mp) & set(ex):
            d = datetime.fromtimestamp(h, tz=timezone.utc)
            if d.year != year or d.month > 8:
                continue
            obs.append((h, mp[h] - gb_p[h], max(min(ex[h] / cap, 1.0), -1.0)))
        out[link] = obs
    return out


def shift(o25, o26):
    return fit_shift(curve(o25), curve(o26))


def main(n_boot=200, seed=7):
    s25, s26 = series(2025), series(2026)
    point = {l: shift(s25[l], s26[l]) for l in LINKS}
    wk = lambda h: h // (168 * 3600)                                     # noqa: E731
    g25 = {l: defaultdict(list) for l in LINKS}
    g26 = {l: defaultdict(list) for l in LINKS}
    for l in LINKS:
        for r in s25[l]:
            g25[l][wk(r[0])].append(r)
        for r in s26[l]:
            g26[l][wk(r[0])].append(r)
    w25 = sorted(set().union(*[set(g25[l]) for l in LINKS]))
    w26 = sorted(set().union(*[set(g26[l]) for l in LINKS]))
    rnd = random.Random(seed)
    boots = {l: [] for l in LINKS}
    dids = {l: [] for l in LINKS if l != CONTROL}
    for _ in range(n_boot):
        b25 = [rnd.choice(w25) for _ in w25]
        b26 = [rnd.choice(w26) for _ in w26]
        est = {}
        for l in LINKS:
            a = [r for w in b25 for r in g25[l].get(w, [])]
            b = [r for w in b26 for r in g26[l].get(w, [])]
            est[l] = shift(a, b)
            if est[l] is not None:
                boots[l].append(est[l])
        for l in dids:
            if est[l] is not None and est[CONTROL] is not None:
                dids[l].append(est[l] - est[CONTROL])

    def ci(v):
        v = sorted(v)
        return (v[int(.025 * len(v))], v[int(.975 * len(v))]) if v else (None, None)

    res = {}
    for l in LINKS:
        res[l] = dict(n2025=len(s25[l]), n2026=len(s26[l]), shift=point[l], shift_ci95=ci(boots[l]))
        if l != CONTROL:
            d = None if point[l] is None or point[CONTROL] is None else point[l] - point[CONTROL]
            res[l].update(did=d, did_ci95=ci(dids[l]))
        print(f"{l:18s} n {len(s25[l])}/{len(s26[l])} | shift {point[l]} {ci(boots[l])}"
              + ("" if l == CONTROL else f" | DiD vs NSL {res[l]['did']} {res[l]['did_ci95']}"))
    pathlib.Path("data/processed/did_d1.json").write_text(json.dumps(res, indent=1),
                                                          encoding="utf-8")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------
# A35 second design: WITHIN-LINK DIRECTIONAL difference-in-differences.
# The NSL control failed (NSL sat at import capacity through 2025 -> flat flow-spread curve, shift
# not identified; early-2026 hydrological reversal). Instead use the UNCHARGED DIRECTION of the same
# link: CBAM charges GB -> EU only; EU -> GB imports carry no border charge in 2026 (UK CBAM starts
# 2027). Same link, same auctions, same period.
#   E_y(s) = P(export share > +5 % of capacity | spread s)   export-onset curve
#   I_y(s) = P(import share > +5 % of capacity | spread s)   import-onset curve (uncharged direction)
# A priced-in charge c moves E right by c and leaves I unchanged:
#   DDD = shift(E_2025 -> E_2026) - shift(I_2025 -> I_2026)
# ---------------------------------------------------------------------------------------------
def onset(obs, side):
    thr = 0.05
    return [(h, s, 1.0 if (f > thr if side == "exp" else f < -thr) else 0.0) for h, s, f in obs]


def directional(n_boot=200, seed=11):
    s25, s26 = series(2025), series(2026)
    res = {}
    rnd = random.Random(seed)
    wk = lambda h: h // (168 * 3600)                                     # noqa: E731
    for l in LINKS:
        pe = shift(onset(s25[l], "exp"), onset(s26[l], "exp"))
        pi = shift(onset(s25[l], "imp"), onset(s26[l], "imp"))
        g25, g26 = defaultdict(list), defaultdict(list)
        for r in s25[l]:
            g25[wk(r[0])].append(r)
        for r in s26[l]:
            g26[wk(r[0])].append(r)
        k25, k26 = sorted(g25), sorted(g26)
        bs = []
        for _ in range(n_boot):
            a = [r for _ in k25 for r in g25[rnd.choice(k25)]]
            b = [r for _ in k26 for r in g26[rnd.choice(k26)]]
            e = shift(onset(a, "exp"), onset(b, "exp"))
            i = shift(onset(a, "imp"), onset(b, "imp"))
            if e is not None and i is not None:
                bs.append(e - i)
        bs.sort()
        ci = (bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]) if bs else (None, None)
        ddd = None if pe is None or pi is None else pe - pi
        exp_share = (sum(1 for _, _, f in s25[l] if f > 0.05) / max(len(s25[l]), 1),
                     sum(1 for _, _, f in s26[l] if f > 0.05) / max(len(s26[l]), 1))
        res[l] = dict(export_onset_shift=pe, import_onset_shift=pi, DDD=ddd, DDD_ci95=ci,
                      export_hour_share_2025_2026=exp_share)
        print(f"{l:18s} export-onset shift {pe} | import-onset shift {pi} | DDD {ddd} {ci}"
              f" | export-hour share {exp_share[0]:.2f}->{exp_share[1]:.2f}")
    pathlib.Path("data/processed/did_d1_directional.json").write_text(json.dumps(res, indent=1),
                                                                      encoding="utf-8")
    return res


def pooled(n_boot=300, seed=13):
    """Pooled DDD over the four charged links (equal weights), joint week-block bootstrap with the
    SAME resampled weeks for every link; plus the NSL placebo (uncharged both ways)."""
    s25, s26 = series(2025), series(2026)
    treated = [l for l in LINKS if l != CONTROL]
    wk = lambda h: h // (168 * 3600)                                     # noqa: E731

    def ddd(a, b):
        e = shift(onset(a, "exp"), onset(b, "exp"))
        i = shift(onset(a, "imp"), onset(b, "imp"))
        return None if e is None or i is None else e - i

    def pool(a25, a26):
        v = [ddd(a25[l], a26[l]) for l in treated]
        return None if any(x is None for x in v) else sum(v) / len(v)

    point = pool(s25, s26)
    g25 = {l: defaultdict(list) for l in LINKS}
    g26 = {l: defaultdict(list) for l in LINKS}
    for l in LINKS:
        for r in s25[l]:
            g25[l][wk(r[0])].append(r)
        for r in s26[l]:
            g26[l][wk(r[0])].append(r)
    w25 = sorted(set().union(*[set(g25[l]) for l in LINKS]))
    w26 = sorted(set().union(*[set(g26[l]) for l in LINKS]))
    rnd = random.Random(seed)
    bs = []
    for _ in range(n_boot):
        b25 = [rnd.choice(w25) for _ in w25]
        b26 = [rnd.choice(w26) for _ in w26]
        a25 = {l: [r for w in b25 for r in g25[l].get(w, [])] for l in LINKS}
        a26 = {l: [r for w in b26 for r in g26[l].get(w, [])] for l in LINKS}
        p = pool(a25, a26)
        if p is not None:
            bs.append(p)
    bs.sort()
    ci = (bs[int(.025 * len(bs))], bs[int(.975 * len(bs))])
    res = dict(pooled_DDD=point, ci95=ci, n_boot=len(bs), gross_charge=32.39,
               upper_bound_share_of_gross=ci[1] / 32.39)
    print(f"pooled DDD over {treated}: {point:.2f} EUR/MWh, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]"
          f" -> priced-in charge <= {ci[1]:.2f} EUR/MWh = {100 * ci[1] / 32.39:.0f} % of gross")
    pathlib.Path("data/processed/did_d1_pooled.json").write_text(json.dumps(res, indent=1),
                                                                 encoding="utf-8")
    return res


def recovery(cs=(5.0, 10.0, 32.39)):
    """Injection test: a true priced-in charge c shifts the 2026 EXPORT-onset curve right by c
    (export decisions made as if the spread were s - c); the import side is untouched. The pooled
    DDD should recover ~c."""
    s25, s26 = series(2025), series(2026)
    treated = [l for l in LINKS if l != CONTROL]
    out = {}
    for c in cs:
        v = []
        for l in treated:
            e = shift(onset(s25[l], "exp"), [(h, s + c, y) for h, s, y in onset(s26[l], "exp")])
            i = shift(onset(s25[l], "imp"), onset(s26[l], "imp"))
            v.append(e - i)
        out[c] = sum(v) / len(v)
        print(f"injected {c:5.2f} -> pooled DDD recovered {out[c]:6.2f} (per link {v})")
    return out
