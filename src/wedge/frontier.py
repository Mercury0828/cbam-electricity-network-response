"""export-side information frontier on GB exports (D-016).

Labels: tier B (dispatch-robust) signs, a MODEL-BASED oracle. Population: tier-B-labelled GB export
MWh pooled over NL, BE, FR - a country-of-origin rule cannot see the destination. Exporter-side
observable classes are coarse and fixed in advance (D-016). Fitted on 2025, evaluated on 2026.

GAP_H: at the helpful volume the default attains (all of it), the harmful MWh no export-side rule
can avoid = harmful MWh in classes that also contain helpful MWh. Share of labelled volume.
"""
from __future__ import annotations
import json, pathlib, random, sys
from collections import defaultdict
from datetime import datetime, timezone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wedge import inputs_gb_fr_2025_12 as G, run_year_gbfr as RG     # noqa: E402
from wedge.run_tiers import run_border, C26, month                    # noqa: E402


def records(year, mm):
    raw = pathlib.Path(f"data/raw/gb_fr_{year}")
    recs = []
    for imp in ("nl", "be", "fr"):
        links, ta, pa, ea = RG.IMPORTERS[imp]
        carbon = ((lambda h, pa=pa: (G.P_CO2_GB_EUR_PER_T, getattr(G, pa))) if year == 2025 else
                  (lambda h: (C26[str(month(h).month)]["gb_dispatch_eur"], C26[str(month(h).month)]["eua_eur"])))
        xg, xp, *_ = RG.load_gb_fr(raw, imp)
        d = run_border(imp, year, mm, lambda imp=imp: RG.load_gb_fr(raw, imp), G.GB_TECHS,
                       getattr(G, ta), carbon, getattr(G, ea), 0.430, -1)
        for h, v in d.items():
            if v["B"] not in (1, -1):
                continue
            tot = sum(max(s.get(h, 0.0), 0.0) for s in xg.values()) or 1.0
            ren = sum(xg.get(k, {}).get(h, 0.0) for k in ("Wind Onshore", "Wind Offshore", "Solar")) / tot
            gas = xg.get("Fossil Gas", {}).get(h, 0.0) / tot
            recs.append(dict(h=h, imp=imp, mwh=v["mwh"], lab=v["B"], price=xp[h], ren=ren, gas=gas))
    return recs


def qcuts(vals, k):
    s = sorted(vals)
    return [s[int(len(s) * i / k)] for i in range(1, k)]


def cls(r, pc, rc, gc):
    b = lambda x, cuts: sum(x > c for c in cuts)
    return (b(r["price"], pc), b(r["ren"], rc), b(r["gas"], gc))


def frontier(recs, key):
    H, M = defaultdict(float), defaultdict(float)
    for r in recs:
        (H if r["lab"] == -1 else M)[key(r)] += r["mwh"]
    tot = sum(H.values()) + sum(M.values())
    gap_h = sum(M[c] for c in set(H) | set(M) if H[c] > 0) / tot if tot else None
    return H, M, tot, gap_h


def gap_ci(recs, key, n=300, seed=0):
    """Week-block bootstrap (168 h, D-009) of GAP_H.

    🔴 A32: the interval is CONDITIONAL on the fixed inputs, the tier-B labels and the declared
    classes. It reflects sampling variation over weeks only; it does not cover source-data failure,
    model or label uncertainty, or the choice of partition."""
    weeks = defaultdict(list)
    for r in recs:
        weeks[r["h"] // (168 * 3600)].append(r)
    ks = list(weeks); rnd = random.Random(seed); out = []
    for _ in range(n):
        smp = [r for _ in ks for r in weeks[rnd.choice(ks)]]
        out.append(frontier(smp, key)[3])
    out.sort()
    return out[int(.025 * n)], out[int(.975 * n)]


def oos(train, test, key):
    """Rule fitted on 2025: charge a class only if its 2025 harmful volume is zero ('clean'
    classes), which preserves 2025-helpful volume without 2025 misdirection. Evaluated on 2026."""
    H1, M1, _, _ = frontier(train, key)
    H2, M2, tot2, _ = frontier(test, key)
    charged = [c for c in set(H2) | set(M2) if M1.get(c, 0) == 0 and H1.get(c, 0) > 0]
    wd = sum(H2[c] for c in charged); md = sum(M2[c] for c in charged)
    return dict(helpful_kept_pct=100 * wd / sum(H2.values()) if H2 else None,
                harmful_charged_pct_of_labelled=100 * md / tot2 if tot2 else None)


if __name__ == "__main__":
    import wedge.reclear as _RC
    INDEP = "--independent" in sys.argv     # A33 (review B3): same labels as tiers_independent
    _RC.INDEPENDENT_SIGN_CORNERS = INDEP
    r25, r26 = records(2025, 12), records(2026, 8)
    pc = qcuts([r["price"] for r in r25], 5)
    rc = qcuts([r["ren"] for r in r25], 3)
    gc = qcuts([r["gas"] for r in r25], 3)
    key = lambda r: cls(r, pc, rc, gc)
    res = {"class_cuts_fixed_on_2025": dict(price=pc, ren=rc, gas=gc)}
    for tag, recs in (("2025", r25), ("2026", r26)):
        H, M, tot, g = frontier(recs, key)
        lo, hi = gap_ci(recs, key)
        default_mis = 100 * sum(M.values()) / tot
        res[tag] = dict(labelled_mwh=tot, default_misdirected_pct=default_mis,
                        GAP_H_pct=100 * g, GAP_H_ci95=[100 * lo, 100 * hi],
                        avoidable_by_exporter_rule_pct=default_mis - 100 * g,
                        classes_used=len(set(H) | set(M)))
        print(f"{tag}: labelled {tot:,.0f} MWh | default (charge all) misdirected {default_mis:5.2f}% | "
              f"GAP_H {100*g:5.2f}% [95% {100*lo:5.2f}, {100*hi:5.2f}] | avoidable by exporter-only rule "
              f"{default_mis-100*g:5.2f} pp | oracle 0.00%")
    o = oos(r25, r26, key)
    res["out_of_sample_2025_to_2026"] = o
    print(f"OOS rule fitted 2025 -> 2026: keeps {o['helpful_kept_pct']:.1f}% of helpful volume, "
          f"charges harmful = {o['harmful_charged_pct_of_labelled']:.2f}% of labelled")
    # 🔴 A29: the ex-ante PRICE-ONLY sensitivity (A24) is now computed here, so a
    # clean rerun regenerates every number in frontier_gb.json. Classes = 2025 price quantiles only.
    # 🔴 A29: GAP_H is a DESCRIPTIVE in-sample overlap for these declared
    # partitions and the objective "keep 100 % of helpful volume"; it is not a bound over every
    # exporter-only rule. The 2025 -> 2026 rule is the separate held-out calculation. Elexon MID is
    # a short-term market index, not a day-ahead auction price, so price classes are not claimed
    # to be decision-time information.
    res["price_only_sensitivity"] = {}
    for k in (5, 10):
        pk = qcuts([r["price"] for r in r25], k)
        kf = lambda r, pk=pk: sum(r["price"] > c for c in pk)
        _, _, _, g26 = frontier(r26, kf)
        lo26, hi26 = gap_ci(r26, kf)
        res["price_only_sensitivity"][f"price_{k}"] = dict(
            GAP_H_2026=100 * g26, ci=[100 * lo26, 100 * hi26], oos=oos(r25, r26, kf))
        print(f"price-only {k} bins: GAP_H 2026 {100*g26:5.2f}% [95% {100*lo26:5.2f}, {100*hi26:5.2f}] | "
              f"OOS {res['price_only_sensitivity'][f'price_{k}']['oos']}")
    print("g* test (GAP_H < 5 pp collapses D2):", {t: ("COLLAPSES" if res[t]['GAP_H_pct'] < 5 else "D2 HAS CONTENT") for t in ("2025", "2026")})
    pathlib.Path("data/processed/frontier_gb_independent.json" if INDEP else
                 "data/processed/frontier_gb.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
