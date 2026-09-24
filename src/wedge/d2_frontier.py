"""D2 as a sharp robust information frontier (D-020, pre-registered 2026-09-21 07:48 UTC).

T(m) = (s - p_x) E_x - (s - p_m) m is tau_ref (A37) with the importer state m = E_m unknown to an
exporter-side oracle that knows the hour, the border, its own efficiency corner qx and the loss
factor eta. For each exporter state, u(m) = proj_A T(m) over the importer corners qm, and

    R_inf = (max_m u - min_m u) / 2        (sharp minimax regret vs the two-sided oracle)

A = [0, inf) (statutory non-negativity) and, as a diagnostic, A = R. Pointwise reporting with
coverage; no population-minimax claim. Destination-blind witness pairs for GB hours exporting to both
NL and BE. Scenarios come from benchmark.collect (keyed: qx, qm, eta, E_x, E_m).
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as G, inputs_rs_hu_2025_12 as R      # noqa: E402
from wedge import run_year as RY, run_year_gbfr as RG                        # noqa: E402
from wedge.benchmark import C26, collect                                    # noqa: E402
from wedge.run_tiers import month                                            # noqa: E402

S_MODES = ("cert", "pm", 150.0, 200.0)


def s_of(r, mode):
    return {"cert": r["P_C"], "pm": r["p_m"]}.get(mode, mode)


def oracle_sets(r, mode, constrained):
    """{(qx, eta): [u(m) for each importer corner]} for one hour."""
    s = s_of(r, mode)
    g = defaultdict(list)
    for qx, qm, eta, ex, im in r["sc"]:
        t = (s - r["p_x_dispatch"]) * ex - (s - r["p_m"]) * im
        g[(qx, eta)].append(max(t, 0.0) if constrained else t)
    return g


def frontier(rows, mode, constrained):
    lab = [r for r in rows if r["sc"] is not None]
    tot = sum(r["mwh"] for r in rows)
    w = sum(r["mwh"] for r in lab)
    lo = hi = mat = 0.0
    for r in lab:
        rs = [(max(u) - min(u)) / 2 for u in oracle_sets(r, mode, constrained).values()]
        lo += r["mwh"] * min(rs)
        hi += r["mwh"] * max(rs)
        mat += r["mwh"] * (max(rs) > 1.0)
    return dict(resolved_pct=100 * w / tot, R_inf_mean=[lo / w, hi / w],
                material_share_pct=100 * mat / w)


def blind_pairs(nl, be, mode):
    """Destination-blind witness: same GB hour and exporter state, NL vs BE oracle-action intervals."""
    bnl = {r["h"]: r for r in nl if r["sc"] is not None}
    bbe = {r["h"]: r for r in be if r["sc"] is not None}
    pair_mwh = sep_mwh = half_d = 0.0
    for h in set(bnl) & set(bbe):
        a, b = oracle_sets(bnl[h], mode, True), oracle_sets(bbe[h], mode, True)
        ds = []
        for k in set(a) & set(b):
            ua, ub = a[k], b[k]
            ds.append(max(0.0, max(min(ua), min(ub)) - min(max(ua), max(ub))))
        if not ds:
            continue
        v = bnl[h]["mwh"] + bbe[h]["mwh"]
        d = min(ds)                                  # conservative over exporter states
        pair_mwh += v
        sep_mwh += v * (d > 0)
        half_d += v * d / 2
    return dict(paired_mwh=pair_mwh, separated_share_pct=100 * sep_mwh / pair_mwh if pair_mwh else None,
                mean_half_separation=half_d / pair_mwh if pair_mwh else None)


def main():
    raw = pathlib.Path("data/raw/gb_fr_2026")

    def gb_prices(h):
        m = C26[str(month(h).month)]
        return dict(p_x_dispatch=m["gb_dispatch_eur"], p_x_uka=m["uka_eur"],
                    p_x_credit_full=m["gb_dispatch_eur"])
    rows = {}
    for imp in ("nl", "be"):
        links, ta, pa, ea = RG.IMPORTERS[imp]
        rows[f"GB->{imp.upper()}"] = collect(lambda imp=imp: RG.load_gb_fr(raw, imp), G.GB_TECHS,
                                             getattr(G, ta), gb_prices, getattr(G, ea), -1)
    rraw = pathlib.Path("data/raw/rs_hu_2026")

    def rl():
        rs, hu, rp, hp, fl, rc, hc = RY.load_rs_hu(rraw)
        return rs, rp, hu, hp, fl, rc, hc
    rows["RS->HU"] = collect(rl, R.RS_TECHS, R.HU_TECHS,
                             lambda h: dict(p_x_dispatch=4.0, p_x_uka=4.0, p_x_credit_full=4.0),
                             R.ETA_LINK_RANGE, +1)
    res = {}
    for mode in S_MODES:
        tag = f"s={mode}" if isinstance(mode, str) else f"s={int(mode)}"
        res[tag] = {}
        for b, rr in rows.items():
            c, u = frontier(rr, mode, True), frontier(rr, mode, False)
            res[tag][b] = dict(constrained=c, unconstrained=u)
            print(f"{tag:7s} {b:7s} resolved {c['resolved_pct']:5.1f}% | R_inf A=[0,inf) "
                  f"[{c['R_inf_mean'][0]:6.2f},{c['R_inf_mean'][1]:6.2f}] material {c['material_share_pct']:5.1f}%"
                  f" | A=R [{u['R_inf_mean'][0]:6.2f},{u['R_inf_mean'][1]:6.2f}] material {u['material_share_pct']:5.1f}%")
        bp = blind_pairs(rows["GB->NL"], rows["GB->BE"], mode)
        res[tag]["destination_blind_GB"] = bp
        print(f"        GB blind pairs: {bp['paired_mwh']:,.0f} MWh | separated {bp['separated_share_pct']}"
              f" % | mean d/2 {bp['mean_half_separation']}")
    c = res["s=cert"]
    ver = {
        "P-D20a (GB constrained mean R_inf < 0.5 at s=P_C)":
            all(c[b]["constrained"]["R_inf_mean"][1] < 0.5 for b in ("GB->NL", "GB->BE")),
        "P-D20b (unconstrained mean R_inf < 2 at s=P_C, all borders)":
            all(c[b]["unconstrained"]["R_inf_mean"][1] < 2 for b in rows),
        "P-D20c (constrained mean R_inf >= 5 at s=200 on >= 1 border)":
            any(res["s=200"][b]["constrained"]["R_inf_mean"][1] >= 5 for b in rows),
    }
    for k, v in ver.items():
        print(f"{k}: {'HOLDS' if v else 'FALSIFIED'}")
    res["preregistered_verdicts"] = ver
    pathlib.Path("data/processed/d2_frontier.json").write_text(json.dumps(res, indent=1),
                                                               encoding="utf-8")


if __name__ == "__main__":
    main()
