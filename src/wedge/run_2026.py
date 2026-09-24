"""Charged-period run: GB -> FR / NL / BE, January-August 2026.

The CBAM charge applies to 2026 imports (certificate SALES are postponed to 2027-02-01, B6, but
the liability accrues). This is therefore the CHARGED period, reported separately from the 2025
retrospective and never summed with it.

Carbon prices are MONTHLY (sources in DATA_SOURCES.md): EUA from
KOBiZE Jan-Jul, August from Minpact with UNVERIFIED comparability; UKA from DESNZ converted at
the BoE monthly rate; no UK-EU ETS linkage in force (so UKA for GB, EUA for the continent).
"""
from __future__ import annotations

import json, pathlib, sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wedge import run_year_gbfr as RG   # noqa: E402
from wedge.carbon import load_c26                                   # noqa: E402

C = load_c26()


def carbon(h):
    m = str(datetime.fromtimestamp(h, tz=timezone.utc).month)
    return C[m]["gb_dispatch_eur"], C[m]["eua_eur"]


rows = []
for imp in ("fr", "nl", "be"):
    for rule in ("marginal_v2", "marginal_v3"):
        r = RG.run(pathlib.Path("data/raw/gb_fr_2026"), f"GB->{imp.upper()} 2026 {rule}",
                   candidate_rule=rule, imp=imp, carbon=carbon, year=2026, max_month=8)
        rows.append(r)
        v = r["validation"]
        print(f"GB->{imp.upper()} 2026 {rule:12s} exp_h={r['export_hours']:4d} "
              f"{r['mwh_total']:>10,.0f} MWh | VALIDATED misdir={v['validated_mwh_misdirected']:>9,.0f} MWh "
              f"(local contrast {v['local_emission_contrast_tco2_misdirected']:+8,.0f} tCO2, gross levy proxy EUR {v['gross_levy_exposure_proxy_eur_misdirected']:>10,.0f})"
              f" | aligned={v['validated_mwh_aligned']:>9,.0f} MWh")
pathlib.Path("data/processed/charged_2026_gb.json").write_text(json.dumps(rows, indent=1),
                                                               encoding="utf-8")
