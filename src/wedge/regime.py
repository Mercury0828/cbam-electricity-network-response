"""Conservative regime classification and the D1 headline.

Both GB links allocate capacity by EXPLICIT auction (BritNed: Empire; Nemo Link: JAO), so hourly
outturn flows are not mechanically set by a price spread and R1 cannot be positively evidenced
from these data. Therefore:

  "R2" SCREEN: flow >= 95 % of the link's nominal capacity AND spread (m - x) > GROSS charge.
     Using the gross charge makes the CHARGE inequality conservative for any Art. 9 deduction
     (net <= gross, law F7); it does NOT make the classification a sufficient condition for zero
     response (see A29 below).
  R0 (unclassified): everything else - never defaulted into R1.
  Hours with no sourced certificate price (2026 Q3) are R0.

🔴 A29. "R2" here is a NEAR-CAPACITY,
POSITIVE-SPREAD SCREEN, not an established zero-response classification: 95 % of rated capacity
is not proof that available capacity binds, and a mixed-stage outturn spread (Elexon MID vs
continental day-ahead) under explicit auctions does not identify the derivative of nominations
with respect to the charge. Its share is therefore reported as a SCREEN share, never as a lower
bound on unresponsiveness. Hours whose inputs are missing (no capacity; no certificate price,
i.e. Jul-Aug 2026) are reported separately as R0-missing, not pooled into "non-R2".
Capacities: BritNed 1000 MW, Nemo Link 1000 MW, GB-FR 4000 MW (IFA 2000 + IFA2 1000 + ElecLink
1000), RS->HU monthly NTC 2025 from EMS Annual Technical Report 2025 p.74 (2026 NTC not found:
RS->HU 2026 regime = R0).
"""
from __future__ import annotations
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wedge import inputs_gb_fr_2025_12 as G, inputs_rs_hu_2025_12 as R      # noqa: E402
from wedge import run_year as RY, run_year_gbfr as RG                        # noqa: E402
from wedge.run_tiers import run_border, C26, month                            # noqa: E402

CAP = {"nl": 1000.0, "be": 1000.0, "fr": 4000.0}
RSHU_NTC_2025 = [432, 371, 645, 200, 277, 520, 587, 671, 620, 607, 800, 800]


def classify(d, cap_fn):
    out = {}
    for h, v in d.items():
        cap = cap_fn(h)
        if cap is None or v["gross_charge"] is None:
            out[h] = "R0m"                   # missing classification input
        elif v["mwh"] >= 0.95 * cap and v["spread"] > v["gross_charge"]:
            out[h] = "R2"
        else:
            out[h] = "R0"
    return out


def report(tag, d, reg):
    tot = sum(v["mwh"] for v in d.values())
    r2 = sum(v["mwh"] for h, v in d.items() if reg[h] == "R2")
    r0m = sum(v["mwh"] for h, v in d.items() if reg[h] == "R0m")
    # A32: missing-input hours are NOT part of the non-R2 denominator.
    nonr2 = {h: v for h, v in d.items() if reg[h] == "R0"}
    nt = sum(v["mwh"] for v in nonr2.values())
    bmis = sum(v["mwh"] for v in nonr2.values() if v["B"] == 1)
    bali = sum(v["mwh"] for v in nonr2.values() if v["B"] == -1)
    bmis_all = sum(v["mwh"] for v in d.values() if v["B"] == 1)
    bmis_r2 = sum(v["mwh"] for h, v in d.items() if v["B"] == 1 and reg[h] == "R2")
    row = dict(border=tag, charged_mwh=tot, R2_mwh=r2, R2_pct=100 * r2 / tot if tot else None,
               R0_missing_mwh=r0m, R0_missing_pct=100 * r0m / tot if tot else None,
               nonR2_mwh=nt, B_mis_nonR2_mwh=bmis, B_mis_nonR2_pct=100 * bmis / nt if nt else None,
               B_ali_nonR2_pct=100 * bali / nt if nt else None,
               B_mis_in_R2_pct_of_B_mis=100 * bmis_r2 / bmis_all if bmis_all else None)
    print(f"{tag:14s} charged {tot:>12,.0f} MWh | near-cap screen {row['R2_pct']:5.1f}% | R0-missing {row['R0_missing_pct']:5.1f}% | "
          f"non-R2: B misdirected {row['B_mis_nonR2_pct'] if row['B_mis_nonR2_pct'] is None else round(row['B_mis_nonR2_pct'],2)}%, "
          f"aligned {row['B_ali_nonR2_pct'] if row['B_ali_nonR2_pct'] is None else round(row['B_ali_nonR2_pct'],2)}% | "
          f"share of B-misdirected that is R2: {row['B_mis_in_R2_pct_of_B_mis'] if row['B_mis_in_R2_pct_of_B_mis'] is None else round(row['B_mis_in_R2_pct_of_B_mis'],1)}%")
    return row


if __name__ == "__main__":
    rows = []
    for year, mm in ((2025, 12), (2026, 8)):
        raw = pathlib.Path(f"data/raw/gb_fr_{year}")
        for imp in ("nl", "be", "fr"):
            links, ta, pa, ea = RG.IMPORTERS[imp]
            carbon = ((lambda h, pa=pa: (G.P_CO2_GB_EUR_PER_T, getattr(G, pa))) if year == 2025 else
                      (lambda h: (C26[str(month(h).month)]["gb_dispatch_eur"], C26[str(month(h).month)]["eua_eur"])))
            d = run_border(imp, year, mm, lambda imp=imp: RG.load_gb_fr(raw, imp), G.GB_TECHS,
                           getattr(G, ta), carbon, getattr(G, ea), 0.430, -1)
            rows.append(dict(year=year, **report(f"{year} GB->{imp.upper()}", d, classify(d, lambda h, imp=imp: CAP[imp]))))
        rraw = pathlib.Path(f"data/raw/rs_hu_{year}")
        rc = ((lambda h: (R.P_CO2_RS_EUR_PER_T, R.P_CO2_HU_EUR_PER_T)) if year == 2025 else
              (lambda h: (4.0, C26[str(month(h).month)]["eua_eur"])))

        def rl(rraw=rraw):
            rs, hu, rp, hp, fl, rcap, hcap = RY.load_rs_hu(rraw)
            return rs, rp, hu, hp, fl, rcap, hcap
        d = run_border("RS->HU", year, mm, rl, R.RS_TECHS, R.HU_TECHS, rc, R.ETA_LINK_RANGE, 1.041, +1)
        capf = (lambda h: float(RSHU_NTC_2025[month(h).month - 1])) if year == 2025 else (lambda h: None)
        rows.append(dict(year=year, **report(f"{year} RS->HU", d, classify(d, capf))))
    pathlib.Path("data/processed/regime.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
