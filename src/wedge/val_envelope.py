"""Validation (d): do the 50 corner scenarios cover the continuous efficiency envelope? (A41)

Merit-order switches can occur INSIDE an efficiency range, so endpoint
evaluation does not by itself prove robustness over the continuous envelope. For a random sample of
hours per border, kappa = -delta_emissions is evaluated on a DENSE grid (5 values each for the gas and
solid efficiency quantiles, independently on each side: 25 x 25 x 2 loss factors = 1,250 scenarios)
and compared with the 50-scenario corner set.

Reported per border: share of sampled hours whose dense-grid kappa interval is wider than the corner
interval by > 0.005 tCO2/MWh; share whose robust sign (all scenarios signed and agreeing) differs;
share where "e_dec above the entire interval" differs; share newly unresolved (an interior scenario
unsigned). Sample: 250 hours per border (seeded), Jan-Jun 2026.
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as G, inputs_rs_hu_2025_12 as R      # noqa: E402
from wedge import run_year as RY, run_year_gbfr as RG                        # noqa: E402
from wedge.carbon import load_c26                                           # noqa: E402
from wedge.reclear import CORNERS, reclear_hour, units_from                  # noqa: E402
from wedge.run_year import daily_ttf_fuel, techs_for                         # noqa: E402
from wedge.run_tiers import month                                            # noqa: E402

C26 = load_c26()
GRID = [{"gas": a, "solid": b} for a in (0, .25, .5, .75, 1) for b in (0, .25, .5, .75, 1)]


def kappa_set(ux_b, um_b, qs_x, qs_m, f, etas):
    vals = []
    for qx in qs_x:
        ux = ux_b(qx)
        for qm in qs_m:
            um = um_b(qm)
            for eta in etas:
                r = reclear_hour(ux, um, flow_mw=f, eta=eta, cap_flow=1e6)
                if not r.solved or r.sign is None:
                    return None
                vals.append(-r.delta_emissions)
    return min(vals), max(vals)


def run(name, loader, X, M, carbon, eta_rng, sign_flow, e_dec, n=250, seed=5):
    xg, xp, mg, mp, flow, xc, mc = loader()
    fuel = daily_ttf_fuel(1.16)
    hours = [h for h in sorted(set(xp) & set(mp) & set(flow))
             if month(h).year == 2026 and month(h).month <= 6 and sign_flow * flow[h] > 0]
    random.Random(seed).shuffle(hours)
    stats = dict(n=0, wider=0, sign_diff=0, above_diff=0, newly_unresolved=0, both_resolved=0,
                 max_widening=0.0)
    for h in hours[:n]:
        f = sign_flow * flow[h]
        gas, coal = fuel(h)
        px, pm = carbon(h)
        xt, mt = techs_for(X, gas, coal), techs_for(M, gas, coal)
        gx = {k: v[h] for k, v in xg.items() if h in v}
        gm = {k: v[h] for k, v in mg.items() if h in v}
        if not gx or not gm:
            continue
        bx = lambda q: units_from(xt, gx, xc, px, clearing_price=xp[h], q=q)     # noqa: E731
        bm = lambda q: units_from(mt, gm, mc, pm, clearing_price=mp[h], q=q)     # noqa: E731
        c = kappa_set(bx, bm, CORNERS, CORNERS, f, eta_rng)
        d = kappa_set(bx, bm, GRID + [None], GRID + [None], f, eta_rng)
        stats["n"] += 1
        if c is not None and d is None:
            stats["newly_unresolved"] += 1
            continue
        if c is None or d is None:
            continue
        stats["both_resolved"] += 1
        widen = max(c[0] - d[0], d[1] - c[1], 0.0)
        stats["max_widening"] = max(stats["max_widening"], widen)
        stats["wider"] += widen > 0.005
        sgn = lambda iv: 1 if iv[0] > 0 else (-1 if iv[1] < 0 else 0)            # noqa: E731
        stats["sign_diff"] += sgn(c) != sgn(d)
        stats["above_diff"] += (e_dec > c[1]) != (e_dec > d[1])
    print(f"{name:7s} sampled {stats['n']} h | both resolved {stats['both_resolved']} | dense grid wider"
          f" (>0.005) {stats['wider']} | max widening {stats['max_widening']:.4f} t/MWh | robust-sign"
          f" changes {stats['sign_diff']} | e_dec-above changes {stats['above_diff']} | newly unresolved"
          f" {stats['newly_unresolved']}")
    return stats


def main():
    raw = pathlib.Path("data/raw/gb_fr_2026")
    res = {}
    for imp in ("nl", "be"):
        links, ta, pa, ea = RG.IMPORTERS[imp]
        res[f"GB->{imp.upper()}"] = run(
            f"GB->{imp.upper()}", lambda imp=imp: RG.load_gb_fr(raw, imp), G.GB_TECHS, getattr(G, ta),
            lambda h: (C26[str(month(h).month)]["gb_dispatch_eur"], C26[str(month(h).month)]["eua_eur"]),
            getattr(G, ea), -1, 0.430)
    rraw = pathlib.Path("data/raw/rs_hu_2026")

    def rl():
        rs, hu, rp, hp, fl, rc, hc = RY.load_rs_hu(rraw)
        return rs, rp, hu, hp, fl, rc, hc
    res["RS->HU"] = run("RS->HU", rl, R.RS_TECHS, R.HU_TECHS,
                        lambda h: (4.0, C26[str(month(h).month)]["eua_eur"]), R.ETA_LINK_RANGE, +1, 1.041)
    pathlib.Path("data/processed/val_envelope.json").write_text(json.dumps(res, indent=1),
                                                                encoding="utf-8")


if __name__ == "__main__":
    main()
