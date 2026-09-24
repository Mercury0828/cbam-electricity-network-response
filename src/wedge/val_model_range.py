"""Scenario range of the redispatch-model counterparts in Fig. 2 (A48, review item on Fig. 2).

For each fixed efficiency/loss scenario, the volume-weighted mean exporter response e_out = E_x * eta and importer
response E_m over resolved hours; the range is the min and max of these means across the 50 scenarios. Adds
e_out_range and e_in_range to the model entries of data/processed/val_regress.json (existing values untouched).
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as G, inputs_rs_hu_2025_12 as R        # noqa: E402
from wedge import run_year as RY, run_year_gbfr as RG                         # noqa: E402
from wedge.benchmark import C26, collect                                     # noqa: E402
from wedge.run_tiers import month                                             # noqa: E402


def ranges(rows):
    lab = [r for r in rows if r["sc"] is not None]
    w = sum(r["mwh"] for r in lab)
    ex, em = defaultdict(float), defaultdict(float)
    for r in lab:
        for qx, qm, eta, e_x, e_m in r["sc"]:
            k = (qx, qm, round(eta, 6))
            ex[k] += r["mwh"] * e_x * eta
            em[k] += r["mwh"] * e_m
    exs = [v / w for v in ex.values()]
    ems = [v / w for v in em.values()]
    return [min(exs), max(exs)], [min(ems), max(ems)], len(ex)


def main():
    p = pathlib.Path("data/processed/val_regress.json")
    v = json.loads(p.read_text(encoding="utf-8"))
    links, ta, pa, ea = RG.IMPORTERS["nl"]

    def prices(h):
        m = C26[str(month(h).month)]
        return dict(p_x_dispatch=m["gb_dispatch_eur"], p_x_uka=m["uka_eur"], p_x_credit_full=m["gb_dispatch_eur"])
    gb = collect(lambda: RG.load_gb_fr(pathlib.Path("data/raw/gb_fr_2026"), "nl"), G.GB_TECHS, getattr(G, ta),
                 prices, getattr(G, ea), -1)

    def rl():
        rs, hu, rp, hp, fl, rc, hc = RY.load_rs_hu(pathlib.Path("data/raw/rs_hu_2026"))
        return rs, rp, hu, hp, fl, rc, hc
    rs = collect(rl, R.RS_TECHS, R.HU_TECHS, lambda h: dict(p_x_dispatch=4.0, p_x_uka=4.0, p_x_credit_full=4.0),
                 R.ETA_LINK_RANGE, +1)
    for key, rows in (("model_GB_NL_2026H1", gb), ("model_RS_HU_2026H1", rs)):
        a, b, n = ranges(rows)
        v[key]["e_out_range"], v[key]["e_in_range"], v[key]["n_scenarios"] = a, b, n
        print(key, v[key])
    p.write_text(json.dumps(v, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
