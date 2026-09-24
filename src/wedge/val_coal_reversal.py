"""Validation (a) of A41: are the robustly misdirected GB->NL hours plausible coal-reversal hours?

Builder for data/processed/val_coal_reversal.json (R1-08: this artefact previously had no script).
Population: 2026 GB->NL tier-B misdirected hours under INDEPENDENT efficiency corners (A30), Jan-Aug.
Per hour: NL coal output and upward headroom (vs the period's maximum observed coal output), NL gas
output, NL day-ahead price, and the SRMC bands of NL coal and gas over the declared efficiency range
(fuel: daily TTF / monthly coal; carbon: monthly EUA; VOM midpoints).
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import wedge.reclear as RC                                             # noqa: E402

RC.INDEPENDENT_SIGN_CORNERS = True

from wedge import inputs_gb_fr_2025_12 as G, run_year_gbfr as RG      # noqa: E402
from wedge.run_tiers import run_border, C26, month                     # noqa: E402
from wedge.run_year import daily_ttf_fuel, techs_for                   # noqa: E402


def main():
    raw = pathlib.Path("data/raw/gb_fr_2026")
    carbon = lambda h: (C26[str(month(h).month)]["gb_dispatch_eur"],   # noqa: E731
                        C26[str(month(h).month)]["eua_eur"])
    d = run_border("nl", 2026, 8, lambda: RG.load_gb_fr(raw, "nl"), G.GB_TECHS, G.NL_TECHS, carbon,
                   G.ETA_LINK_RANGE_NL, 0.430, -1)
    mis = [h for h, v in d.items() if v["B"] == 1]
    xg, xp, mg, mp, flow, xc, mc = RG.load_gb_fr(raw, "nl")
    coal, gas = mg["Fossil hard coal"], mg["Fossil gas"]
    cmax = max(coal.values())
    fuel = daily_ttf_fuel(1.16)
    rows = []
    for h in mis:
        g, c = fuel(h)
        pm = C26[str(month(h).month)]["eua_eur"]
        t = techs_for(G.NL_TECHS, g, c)

        def srmc(k, q):
            tech = t[k]
            eta = tech.eta_range[0] + q * (tech.eta_range[1] - tech.eta_range[0])
            ef = tech.ef_el_range[1] - q * (tech.ef_el_range[1] - tech.ef_el_range[0])
            return tech.fuel_th / eta + pm * ef + sum(tech.vom_range) / 2
        rows.append(dict(h=h, mwh=-flow[h], coal=coal.get(h, 0.0), coal_headroom=cmax - coal.get(h, 0.0),
                         gas=gas.get(h, 0.0), p_nl=mp[h], ttf=g,
                         coal_srmc=(srmc("Fossil hard coal", 1), srmc("Fossil hard coal", 0)),
                         gas_srmc=(srmc("Fossil gas", 1), srmc("Fossil gas", 0))))
    summary = dict(
        hours=len(rows), mwh=sum(r["mwh"] for r in rows),
        coal_running=sum(r["coal"] > 100 for r in rows),
        headroom_gt_100=sum(r["coal_headroom"] > 100 for r in rows),
        coal_cheaper_worst_case=sum(r["coal_srmc"][1] < r["gas_srmc"][0] for r in rows),
        price_within_coal_band_pm5=sum(r["coal_srmc"][0] - 5 <= r["p_nl"] <= r["coal_srmc"][1] + 5 for r in rows),
        median_price=st.median(r["p_nl"] for r in rows),
        median_coal_band=[st.median(r["coal_srmc"][i] for r in rows) for i in (0, 1)],
        median_gas_band=[st.median(r["gas_srmc"][i] for r in rows) for i in (0, 1)],
        by_month=dict(Counter(month(r["h"]).month for r in rows)))
    print(summary)
    pathlib.Path("data/processed/val_coal_reversal.json").write_text(
        json.dumps(dict(summary=summary, hours=rows), indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
