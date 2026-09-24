"""D1 as model-conditional R2 certificates (D-021 + addendum, pre-registered before computation).

Market certificate (per link-hour): JAO day-ahead GB->EU capacity fully allocated
(allocated >= offered - 1 MW, offered > 0) at a positive clearing price A_h -> locally insulated for
charge changes up to A_h (breakpoint distance A_h).

Model certificate (per border-hour, every one of the 50 independent scenarios):
  1. realised export >= 98 % of available capacity; available = JAO daily offered + long-term
     allocated (monthly products delivering that hour);
  2. lambda = SRMC(importer responder) - SRMC(exporter responder)/eta > c_bar;
  3. realised spread p_m - p_GB/eta > c_bar.
Responsive certificate: export <= 90 % of available AND lambda < G in every scenario.
c_bar in {G, 2G}, G = certificate price x 0.430 (Q1 75.36 / Q2 75.28). Jan-Jun 2026.
Headline lower bound = volume with BOTH certificates / all charged export volume of the border.
BritNed (NL) has no JAO daily data -> unclassified (D-021).
"""
from __future__ import annotations

import glob
import json
import pathlib
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as G_                                # noqa: E402
from wedge import run_year_gbfr as RG                                        # noqa: E402
from wedge.carbon import load_c26                                           # noqa: E402
from wedge.cap_price import load_prices                                     # noqa: E402
from wedge.did_d1 import flows                                              # noqa: E402
from wedge.reclear import CORNERS, reclear_hour, units_from                  # noqa: E402
from wedge.run_year import daily_ttf_fuel, techs_for                         # noqa: E402

C26 = load_c26()
CERT = {1: 75.36, 2: 75.28}
E_DEC = 0.430
BLOCK = re.compile(r"B(\d{2})")
# link -> (JAO prefix, EU code, Elexon interconnector names, model border or None)
LINKS = {
    "Nemo (BE)": ("NLL", "BE", {"Belgium (Nemolink)"}, "be"),
    "IFA (FR)": ("IF1", "FR", {"France(IFA)"}, "fr"),
    "IFA2 (FR)": ("IF2", "FR", {"IFA2 (INTIFA2)"}, "fr"),
    "ElecLink (FR)": ("EL1", "FR", {"Eleclink (INTELEC)"}, "fr"),
    "Viking (DK1)": ("VKL", "D1", {"Denmark (Viking link)"}, None),
}


def month(h):
    return datetime.fromtimestamp(h, tz=timezone.utc)


def gross(h):
    d = month(h)
    return CERT[(d.month - 1) // 3 + 1] * E_DEC


def lt_allocated(prefix, eu):
    """{utc_hour: MW} long-term (Monthly horizon) capacity allocated GB->EU for each delivery hour."""
    out = defaultdict(float)
    for f in sorted(glob.glob(f"data/raw/jao/jao_Monthly_{prefix}-GB-{eu}_*.json")):
        for a in json.loads(pathlib.Path(f).read_text(encoding="utf-8")):
            if a.get("cancelled"):
                continue
            t0 = datetime.fromisoformat(a["marketPeriodStart"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(a["marketPeriodStop"].replace("Z", "+00:00"))
            alloc = sum(float(r.get("allocatedCapacity") or 0.0) for r in a.get("results") or [])
            t = t0
            while t < t1:
                out[int(t.timestamp())] += alloc
                t += timedelta(hours=1)
    return out


def model_lambda(border):
    """{h: (min lambda over scenarios, max lambda, eta_mid)} for GB -> border, Jan-Jun 2026, and
    realised spread. None when any scenario is unsolved / unsigned (intertemporal, tie)."""
    links, ta, pa, ea = RG.IMPORTERS[border]
    xg, xp, mg, mp, flow, xc, mc = RG.load_gb_fr(pathlib.Path("data/raw/gb_fr_2026"), border)
    fuel = daily_ttf_fuel(1.16)
    X, M, eta_rng = G_.GB_TECHS, getattr(G_, ta), getattr(G_, ea)
    out = {}
    for h in sorted(set(xp) & set(mp) & set(flow)):
        d = month(h)
        if d.year != 2026 or d.month > 6 or flow[h] >= 0:
            continue
        f = -flow[h]
        px, pm = C26[str(d.month)]["gb_dispatch_eur"], C26[str(d.month)]["eua_eur"]
        gas, coal = fuel(h)
        xt, mt = techs_for(X, gas, coal), techs_for(M, gas, coal)
        gx = {k: v[h] for k, v in xg.items() if h in v}
        gm = {k: v[h] for k, v in mg.items() if h in v}
        if not gx or not gm:
            continue
        lams, ok = [], True
        for qx in CORNERS:
            ux = units_from(xt, gx, xc, px, clearing_price=xp[h], q=qx)
            sx = {u.name: u.srmc for u in ux}
            for qm in CORNERS:
                um = units_from(mt, gm, mc, pm, clearing_price=mp[h], q=qm)
                sm = {u.name: u.srmc for u in um}
                for eta in eta_rng:
                    r = reclear_hour(ux, um, flow_mw=f, eta=eta, cap_flow=1e6)
                    if not r.solved or r.sign is None or not r.exporter_responder \
                            or not r.importer_responder:
                        ok = False
                        break
                    lams.append(sm[r.importer_responder] - sx[r.exporter_responder] / eta)
                if not ok:
                    break
            if not ok:
                break
        out[h] = None if not ok else (min(lams), max(lams), sum(eta_rng) / 2,
                                      mp[h] - xp[h] / (sum(eta_rng) / 2))
    return out


def main():
    lam_cache = {}
    res = {}
    for link, (prefix, eu, names, border) in LINKS.items():
        cp = load_prices(prefix, eu, "exp")                     # {h: (price, offered, allocated)}
        lt = lt_allocated(prefix, eu)
        fl = flows(2026, names)                                 # + = export from GB
        if border and border not in lam_cache:
            lam_cache[border] = model_lambda(border)
        lam = lam_cache.get(border, {})
        tot = mkt = mod = both = both2 = resp = 0.0
        a_dist = []
        n_hours = 0
        for h, q in fl.items():
            d = month(h)
            if d.year != 2026 or d.month > 6 or q <= 0:
                continue
            n_hours += 1
            tot += q
            G = gross(h)
            c = cp.get(h)
            market = bool(c and c[1] and c[1] > 0 and c[2] is not None
                          and c[2] >= c[1] - 1.0 and c[0] > 0)
            if market:
                mkt += q
                a_dist.append((c[0], q))
            model = model2 = responsive = False
            L = lam.get(h) if border else None
            if c and c[1] is not None and L:
                avail = (c[1] or 0.0) + lt.get(h, 0.0)
                at_cap = avail > 0 and q >= 0.98 * avail
                interior = avail > 0 and q <= 0.90 * avail
                model = at_cap and L[0] > G and L[3] > G
                model2 = at_cap and L[0] > 2 * G and L[3] > 2 * G
                responsive = interior and L[1] < G
            if model:
                mod += q
            if model and market:
                both += q
            if model2 and market:
                both2 += q
            if responsive:
                resp += q
        a_dist.sort()
        cum, med = 0.0, None
        for a, q in a_dist:
            cum += q
            if med is None and cum >= 0.5 * mkt:
                med = a
        share_ge_G = (sum(q for a, q in a_dist if a >= 32.39) / mkt) if mkt else None
        res[link] = dict(export_hours=n_hours, export_mwh=tot,
                         market_cert_pct=100 * mkt / tot if tot else None,
                         model_cert_pct=100 * mod / tot if tot else None,
                         both_pct=100 * both / tot if tot else None,
                         both_2G_pct=100 * both2 / tot if tot else None,
                         responsive_pct=100 * resp / tot if tot else None,
                         market_cert_median_breakpoint_eur=med,
                         market_cert_share_breakpoint_ge_G=share_ge_G)
        r = res[link]
        print(f"{link:14s} export {tot:>11,.0f} MWh ({n_hours} h) | market cert {r['market_cert_pct']:.1f}%"
              f" (median breakpoint A_h {med}, >=G {share_ge_G}) | model cert {r['model_cert_pct']}"
              f" | BOTH {r['both_pct']} | both@2G {r['both_2G_pct']} | responsive {r['responsive_pct']}")
    ver = any((v["both_pct"] or 0) > 0 or (v["market_cert_pct"] or 0) > 0 for v in res.values())
    res["P-D21a"] = "HOLDS" if any((v["both_pct"] or 0) > 0 for v in res.values()
                                   if isinstance(v, dict)) else "FALSIFIED"
    print("P-D21a (pi_R2 > 0 on at least one JAO link, model AND market):", res["P-D21a"])
    pathlib.Path("data/processed/r2_cert.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
