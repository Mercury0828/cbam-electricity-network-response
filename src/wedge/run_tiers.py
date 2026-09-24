"""Three evidence tiers per border-hour, both periods, every admitted border (A20-A22).

  A  VALIDATED       mefband determines AND reclear agrees AND coverage holds (D-014). Strict.
  B  DISPATCH-ROBUST reclear sign identical at EVERY declared corner of the pre-registered
                     efficiency axis AND at both loss-factor bounds. Model-based.
  C  DISPATCH-POINT  reclear sign at the midpoints. Model-based, least robust.

Tiers are nested in robustness, NEVER summed, and every share keeps its full denominator.
Also reported: D2 same-hour opposite signs across GB destinations, and, for tier-B misdirected
volume, the share whose price spread exceeds the gross charge (a congestion/R2 flag - NOT a
regime classification, which needs NTC and stage-consistent prices).
"""
from __future__ import annotations
import json, pathlib, sys
from collections import defaultdict
from datetime import datetime, timezone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wedge import inputs_gb_fr_2025_12 as G, inputs_rs_hu_2025_12 as R      # noqa: E402
from wedge import run_year as RY, run_year_gbfr as RG                        # noqa: E402
from wedge.run_year import daily_ttf_fuel, obs, techs_for                    # noqa: E402
from wedge.mefband import Direction, build_candidates, sign_from_sets        # noqa: E402
from wedge.reclear import units_from, reclear_hour, check_against_mefband, robust_sign  # noqa: E402
from wedge.carbon import load_c26                                   # noqa: E402

C26 = load_c26()
CERT = {1: 75.36, 2: 75.28}


def month(h): return datetime.fromtimestamp(h, tz=timezone.utc)


def gross_charge(h, e_dec, eua_fallback):
    d = month(h)
    cp = CERT.get((d.month - 1) // 3 + 1) if d.year == 2026 else eua_fallback
    return None if cp is None else cp * e_dec


def run_border(name, year, mm, loader, X, M, carbon, eta_rng, e_dec, sign_flow):
    xg, xp, mg, mp, flow, xc, mc = loader()
    fuel = daily_ttf_fuel(1.16)
    out = {}
    for h in sorted(set(xp) & set(mp) & set(flow)):
        d = month(h)
        if d.year != year or d.month > mm:
            continue
        f = sign_flow * flow[h]
        if f <= 0:
            continue
        gas, coal = fuel(h); px, pm = carbon(h)
        xt, mt = techs_for(X, gas, coal), techs_for(M, gas, coal)
        gx = {k: v[h] for k, v in xg.items() if h in v}
        gm = {k: v[h] for k, v in mg.items() if h in v}
        if not gx or not gm:
            continue
        cx = build_candidates("x", obs(xg, xc, xt, h), clearing_price=xp[h], p_co2=px,
                              direction=Direction.REDUCE, price_tolerance=G.price_tolerance(xp[h]),
                              candidate_rule="marginal_v2")
        cm = build_candidates("m", obs(mg, mc, mt, h), clearing_price=mp[h], p_co2=pm,
                              direction=Direction.INCREASE, price_tolerance=G.price_tolerance(mp[h]),
                              candidate_rule="marginal_v2")
        s, _ = sign_from_sets(cx, cm, eta_rng)
        mid = reclear_hour(units_from(xt, gx, xc, px, clearing_price=xp[h]),
                           units_from(mt, gm, mc, pm, clearing_price=mp[h]),
                           flow_mw=f, eta=sum(eta_rng) / 2, cap_flow=1e6)
        chk = check_against_mefband(mid, s, cx, cm)
        rb, _ = robust_sign(lambda q: units_from(xt, gx, xc, px, clearing_price=xp[h], q=q),
                            lambda q: units_from(mt, gm, mc, pm, clearing_price=mp[h], q=q),
                            flow_mw=f, eta_range=eta_rng)
        gc = gross_charge(h, e_dec, pm)
        out[h] = dict(mwh=f, A=(s if chk.get("agreement") and chk.get("coverage_ok") else None),
                      B=rb, C=mid.sign if mid.solved else None, dE=mid.delta_emissions,
                      spread=mp[h] - xp[h], gross_charge=gc)
    return out


def summarise(tag, d):
    tot = sum(v["mwh"] for v in d.values())
    row = {"border": tag, "mwh_total": tot}
    for t in "ABC":
        mis = sum(v["mwh"] for v in d.values() if v[t] == 1)
        ali = sum(v["mwh"] for v in d.values() if v[t] == -1)
        row[t] = dict(mis=mis, ali=ali, mis_pct=100 * mis / tot if tot else None,
                      ali_pct=100 * ali / tot if tot else None,
                      resolved_pct=100 * (mis + ali) / tot if tot else None,
                      mis_contrast_tco2=sum(v["dE"] * v["mwh"] for v in d.values() if v[t] == 1))
    bm = [v for v in d.values() if v["B"] == 1 and v["gross_charge"] is not None]
    row["B_mis_spread_gt_charge_pct"] = (100 * sum(v["mwh"] for v in bm if abs(v["spread"]) > v["gross_charge"])
                                         / sum(v["mwh"] for v in bm)) if bm else None
    return row


if __name__ == "__main__":
    import wedge.reclear as _RC
    INDEP = "--independent" in sys.argv          # A30 sensitivity: independent efficiency corners
    _RC.INDEPENDENT_SIGN_CORNERS = INDEP
    rows, by = [], {}
    for year, mm in ((2025, 12), (2026, 8)):
        gbraw = pathlib.Path(f"data/raw/gb_fr_{year}")
        for imp in ("nl", "be", "fr"):
            links, ta, pa, ea = RG.IMPORTERS[imp]
            if year == 2025:
                carbon = lambda h, pa=pa: (G.P_CO2_GB_EUR_PER_T, getattr(G, pa))
            else:
                carbon = lambda h: (C26[str(month(h).month)]["gb_dispatch_eur"], C26[str(month(h).month)]["eua_eur"])
            d = run_border(f"GB->{imp.upper()}", year, mm, lambda imp=imp: RG.load_gb_fr(gbraw, imp),
                           G.GB_TECHS, getattr(G, ta), carbon, getattr(G, ea), 0.430, -1)
            by[(year, imp)] = d
            rows.append(dict(year=year, **summarise(f"GB->{imp.upper()}", d)))
        rraw = pathlib.Path(f"data/raw/rs_hu_{year}")
        rcarbon = ((lambda h: (R.P_CO2_RS_EUR_PER_T, R.P_CO2_HU_EUR_PER_T)) if year == 2025 else
                   (lambda h: (4.0, C26[str(month(h).month)]["eua_eur"])))

        def rl(rraw=rraw):
            rs, hu, rp, hp, fl, rc, hc = RY.load_rs_hu(rraw)
            return rs, rp, hu, hp, fl, rc, hc
        d = run_border("RS->HU", year, mm, rl, R.RS_TECHS, R.HU_TECHS, rcarbon, R.ETA_LINK_RANGE, 1.041, +1)
        rows.append(dict(year=year, **summarise("RS->HU", d)))
        # D2 at tier B: same GB hour, opposite robust signs across destinations
        hs = defaultdict(dict)
        for imp in ("nl", "be", "fr"):
            for h, v in by[(year, imp)].items():
                if v["B"] in (1, -1):
                    hs[h][imp] = (v["B"], v["mwh"])
        opp = {h: x for h, x in hs.items() if len({s for s, _ in x.values()}) == 2}
        rows.append(dict(year=year, border="D2-tierB", multi_dest_hours=sum(1 for x in hs.values() if len(x) >= 2),
                         opposite_hours=len(opp), opposite_mwh=sum(m for x in opp.values() for _, m in x.values())))

    for r in rows:
        if r["border"] == "D2-tierB":
            print(f"{r['year']} D2 tier B: multi-destination hours {r['multi_dest_hours']}, OPPOSITE-SIGN hours {r['opposite_hours']} ({r['opposite_mwh']:,.0f} MWh)")
            continue
        a, b, c = r["A"], r["B"], r["C"]
        print(f"{r['year']} {r['border']:7s} {r['mwh_total']:>12,.0f} MWh | "
              f"A mis {a['mis_pct']:5.2f}% ali {a['ali_pct']:5.2f}% | "
              f"B mis {b['mis_pct']:5.2f}% ali {b['ali_pct']:5.2f}% (resolved {b['resolved_pct']:5.1f}%) | "
              f"C mis {c['mis_pct']:5.2f}% | B-mis spread>charge {r['B_mis_spread_gt_charge_pct'] if r['B_mis_spread_gt_charge_pct'] is None else round(r['B_mis_spread_gt_charge_pct'],1)}%")
    pathlib.Path("data/processed/tiers_independent.json" if INDEP else
                 "data/processed/tiers.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
