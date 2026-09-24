"""What drives tier-B (dispatch-robust) misdirection? Responder pairs, by border and year."""
from __future__ import annotations
import json, pathlib, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wedge import inputs_gb_fr_2025_12 as G, run_year_gbfr as RG          # noqa: E402
from wedge.run_year import daily_ttf_fuel, techs_for                       # noqa: E402
from wedge.reclear import units_from, reclear_hour, robust_sign            # noqa: E402
from wedge.carbon import load_c26                                   # noqa: E402

C26 = load_c26()
fuel = daily_ttf_fuel(1.16)
for year, mm in ((2025, 12), (2026, 8)):
    raw = pathlib.Path(f"data/raw/gb_fr_{year}")
    for imp in ("nl", "be"):
        links, ta, pa, ea = RG.IMPORTERS[imp]
        xg, xp, mg, mp, flow, xc, mc = RG.load_gb_fr(raw, imp)
        pairs, gasb = Counter(), defaultdict(float)
        tot = 0.0
        for h in sorted(set(xp) & set(mp) & set(flow)):
            d = datetime.fromtimestamp(h, tz=timezone.utc)
            if d.year != year or d.month > mm or flow[h] >= 0:
                continue
            f = -flow[h]; gas, coal = fuel(h)
            px, pm = ((G.P_CO2_GB_EUR_PER_T, getattr(G, pa)) if year == 2025 else
                      (C26[str(d.month)]["gb_dispatch_eur"], C26[str(d.month)]["eua_eur"]))
            xt, mt = techs_for(G.GB_TECHS, gas, coal), techs_for(getattr(G, ta), gas, coal)
            gx = {k: v[h] for k, v in xg.items() if h in v}; gm = {k: v[h] for k, v in mg.items() if h in v}
            if not gx or not gm:
                continue
            tot += f
            rb, _ = robust_sign(lambda q: units_from(xt, gx, xc, px, clearing_price=xp[h], q=q),
                                lambda q: units_from(mt, gm, mc, pm, clearing_price=mp[h], q=q),
                                flow_mw=f, eta_range=getattr(G, ea))
            if rb != 1:
                continue
            r = reclear_hour(units_from(xt, gx, xc, px, clearing_price=xp[h]),
                             units_from(mt, gm, mc, pm, clearing_price=mp[h]),
                             flow_mw=f, eta=sum(getattr(G, ea)) / 2, cap_flow=1e6)
            key = (r.exporter_tie or (r.exporter_responder,), r.importer_tie or (r.importer_responder,))
            pairs[(tuple(sorted(set(n.split('_',1)[1] for n in key[0]))), tuple(sorted(set(n.split('_',1)[1] for n in key[1]))))] += f
            gasb["TTF>=44" if gas >= 44 else "TTF<44"] += f
        mis = sum(pairs.values())
        print(f"{year} GB->{imp.upper()}: tier-B misdirected {mis:,.0f} MWh of {tot:,.0f}")
        for k, v in pairs.most_common(5):
            print(f"     {str(k[0]):38s} -> {str(k[1]):22s} {v:>10,.0f} MWh ({100*v/mis:4.1f}%)")
        print("     by gas regime:", {k: f"{100*v/mis:.1f}%" for k, v in gasb.items()})
