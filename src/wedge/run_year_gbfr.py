"""Full-year identification + re-clearing validation for GB -> FR.

Mirrors `run_year.py` (RS -> HU) with the GB/FR data layout: GB from Elexon in weekly
chunks, FR from Energy-Charts in monthly chunks. Same frozen rule (D-009, D-013), same
corrections (A8 nuclear, A10 variable renewables, A12 validation instrument).

Two GB-specific traps, both already hit once and guarded here:
  * Elexon prices are GBP/MWh -> converted to EUR at the BoE rate (B18).
  * UK ETS and EU ETS were NOT linked in 2025 -> UKA for GB, EUA for FR.

🔴 Carbon prices are the December-2025 values for the whole year pending monthly series.
That is a declared simplification and a sensitivity axis, not a claim.
"""

from __future__ import annotations

import glob
import json
import pathlib
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wedge import inputs_gb_fr_2025_12 as INP          # noqa: E402
from wedge.mefband import Direction, build_candidates, sign_from_sets  # noqa: E402
from wedge.reclear import check_against_mefband, reclear_hour, units_from  # noqa: E402
from wedge.run_year import _merge, daily_ttf_fuel, obs, techs_for  # noqa: E402

FR_LINKS = {"France(IFA)", "IFA2 (INTIFA2)", "Eleclink (INTELEC)"}

# Declared electricity default for GB imports, Reg (EU) 2025/2621 Annex III (law/default_values.csv).
E_DEC_GB = 0.430   # tCO2eq / MWh

# CBAM certificate price for 2026 imports: QUARTERLY volume-weighted average of EU ETS auction
# clearing prices (Art. 21(1a), Impl. Reg. 2025/2548), published by the Commission:
#   Q1 2026 75.36, Q2 2026 75.28 EUR/certificate; Q3 not yet published (due 2026-10-05).
# Electricity free-allocation adjustment = 0 (Impl. Reg. 2025/2620 Art. 1(2)).
# Art. 9 deduction for default-value electricity needs a yearly default carbon price that has
# NOT been published for the UK or Serbia -> the NET charge is only bounded, [0, gross].
CBAM_CERT_PRICE_2026 = {1: 75.36, 2: 75.28}          # by quarter


def cert_price(h, fallback):
    d = datetime.fromtimestamp(h, tz=timezone.utc)
    if d.year == 2026:
        return CBAM_CERT_PRICE_2026.get((d.month - 1) // 3 + 1)   # None for Q3: unsourced
    return fallback              # 2025 is retrospective: no certificate price existed

# GB importer zones admitted by the frozen border selection rule. (code, links, techs, p_co2, eta)
IMPORTERS = {
    "fr": (FR_LINKS, "FR_TECHS", "P_CO2_FR_EUR_PER_T", "ETA_LINK_RANGE"),
    "nl": ({"Netherlands(BritNed)"}, "NL_TECHS", "P_CO2_NL_EUR_PER_T", "ETA_LINK_RANGE_NL"),
    "be": ({"Belgium (Nemolink)"}, "BE_TECHS", "P_CO2_BE_EUR_PER_T", "ETA_LINK_RANGE_BE"),
}


def _hour(iso: str) -> int:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    return int(dt.replace(minute=0, second=0, microsecond=0).timestamp())


# 🔴 A28. The per-type actual-generation feed
# (B1620) has publisher DROPOUTS: whole conventional blocks reported as 0 while wind and solar
# remain (e.g. 2026-02-24 10:00Z gas 0 / nuclear 0; FUELHH CCGT 13,181 MW, NUCLEAR 3,952 MW);
# 823 half-hours in 2025, 1,408 in Jan-Aug 2026. On clean periods FUELHH (settlement metering)
# agrees with B1620 (gas/nuclear/biomass corr >= 0.997, equal means). A half-hour whose gas,
# nuclear or biomass deviates from FUELHH by > max(500 MW, 20 %) has its conventional block
# REPLACED from FUELHH; pumped storage is clipped at 0 (FUELHH is net of pumping). Half-hours
# with B1620 wind = 0 while FUELHH WIND > 1 GW are EXCLUDED (embedded wind is not in FUELHH).
FUELHH_MAP = {"Fossil Gas": ("CCGT", "OCGT"), "Nuclear": ("NUCLEAR",), "Biomass": ("BIOMASS",),
              "Fossil Hard coal": ("COAL",), "Fossil Oil": ("OIL",),
              "Hydro Pumped Storage": ("PS",), "Hydro Run-of-river and poundage": ("NPSHYD",),
              "Other": ("OTHER",)}
CHECK_TYPES = ("Fossil Gas", "Nuclear", "Biomass")
REPAIR_LOG: dict[str, dict] = {}      # str(raw) -> repair counts and flagged hours


_FX26 = None


def _eur_per_gbp(h: int) -> float:
    """🔴 A29: monthly BoE rate for 2026 (carbon_monthly_2026.json, sourced in
    round carbon-prices-2026); the December-2025 constant remains a DECLARED proxy for 2025."""
    global _FX26
    d = datetime.fromtimestamp(h, tz=timezone.utc)
    if d.year == 2026:
        if _FX26 is None:
            _FX26 = json.loads(pathlib.Path("data/processed/carbon_monthly_2026.json")
                               .read_text(encoding="utf-8"))
        m = _FX26.get(str(d.month))
        if m is not None:
            return m["gbp_eur"]
    return INP.GBP_PER_EUR_2025_12


def _fuelhh(raw: pathlib.Path) -> dict[str, dict[str, float]]:
    fu: dict[str, dict[str, float]] = {}
    for f in sorted(glob.glob(str(raw / "gb_fuelhh_*.json"))):
        for rec in json.loads(pathlib.Path(f).read_text(encoding="utf-8")).get("data", []):
            fu.setdefault(rec["startTime"], {})[rec["fuelType"]] = float(rec["generation"])
    return fu


def _fuel_sum(F: dict[str, float], k: str) -> float:
    return sum(F.get(z, 0.0) for z in FUELHH_MAP[k])


def load_gb_fr(raw: pathlib.Path, imp: str = "fr", repair: str = "fuelhh"):
    """repair: 'fuelhh' (default, A28) | 'exclude' (drop every flagged hour; sensitivity) |
    'none' (pre-A28 behaviour incl. flow double counting; reproduction only)."""
    fu = _fuelhh(raw) if repair != "none" else {}
    if repair != "none" and not fu:
        raise FileNotFoundError(f"A28: no FUELHH files in {raw}; run src/wedge/fetch/gb_fuelhh.py")
    half: dict[str, dict[str, float]] = {}
    for f in sorted(glob.glob(str(raw / "gb_generation_*.json"))):
        for rec in json.loads(pathlib.Path(f).read_text(encoding="utf-8")).get("data", []):
            half[rec["startTime"]] = {row["psrType"]: float(row["quantity"])
                                      for row in rec["data"]}   # duplicates are identical
    flagged: set[int] = set()
    excluded_hours: set[int] = set()   # A32: an hour with ANY excluded half-hour is dropped whole
    n_rep = n_exc = 0
    gen: dict[str, dict[int, list[float]]] = {}
    for t, d in half.items():
        h = _hour(t)
        if repair != "none":
            F = fu.get(t)
            wind0 = (all(k in d for k in ("Wind Onshore", "Wind Offshore"))
                     and d["Wind Onshore"] + d["Wind Offshore"] == 0
                     and F is not None and F.get("WIND", 0.0) > 1000)
            if F is None or wind0:
                n_exc += 1
                flagged.add(h)
                excluded_hours.add(h)
                continue
            if any(abs(d.get(k, 0.0) - _fuel_sum(F, k)) > max(500.0, 0.2 * _fuel_sum(F, k))
                   for k in CHECK_TYPES):
                n_rep += 1
                flagged.add(h)
                if repair == "exclude":
                    continue
                d = dict(d)
                for k in FUELHH_MAP:
                    d[k] = max(_fuel_sum(F, k), 0.0)
            # 🔴 A32: validity screen AFTER repair. If the conventional block is
            # still all zero, both sources dropped out together (e.g. 2026-07-07 10:00Z): exclude.
            if all(d.get(k, 0.0) <= 0.0 for k in CHECK_TYPES):
                n_exc += 1
                flagged.add(h)
                excluded_hours.add(h)
                continue
        for k, v in d.items():
            gen.setdefault(k, {}).setdefault(h, []).append(v)
    gb_gen = {k: {h: sum(v) / len(v) for h, v in dd.items()} for k, dd in gen.items()}
    drop = flagged if repair == "exclude" else excluded_hours
    if drop:
        gb_gen = {k: {h: v for h, v in dd.items() if h not in drop}
                  for k, dd in gb_gen.items()}
    REPAIR_LOG[str(raw)] = dict(mode=repair, repaired_halfhours=n_rep,
                                excluded_halfhours=n_exc, flagged_hours=sorted(flagged))

    pr: dict[int, list[float]] = {}
    for f in sorted(glob.glob(str(raw / "gb_price_*.json"))):
        for rec in json.loads(pathlib.Path(f).read_text(encoding="utf-8")).get("data", []):
            if rec.get("dataProvider") == "APXMIDP" and rec.get("volume", 0) > 0:
                # 🔴 B18: GBP -> EUR
                h = _hour(rec["startTime"])
                pr.setdefault(h, []).append(float(rec["price"]) * _eur_per_gbp(h))
    gb_price = {h: sum(v) / len(v) for h, v in pr.items()}

    # 🔴 A28: weekly downloads share boundary dates, so identical records
    # appear twice (2026: 133,420 records, 117,100 unique). DEDUPLICATE on (startTime,
    # interconnector) before aggregating; the pre-A28 loader summed the duplicates.
    rows = [((rec["startTime"], rec["interconnectorName"]), float(rec["generation"]))
            for f in sorted(glob.glob(str(raw / "gb_interconnector_*.json")))
            for rec in json.loads(pathlib.Path(f).read_text(encoding="utf-8")).get("data", [])
            if rec.get("interconnectorName") in IMPORTERS[imp][0]]
    if repair != "none":
        rows = list(dict(rows).items())
    per: dict[int, dict[str, float]] = {}
    for (t, name), v in rows:
        d = per.setdefault(_hour(t), {})
        d[name] = d.get(name, 0.0) + v / 2.0              # two settlement periods per hour
    flow = {h: sum(d.values()) for h, d in per.items()}   # + = import INTO GB

    fr_gen = _merge(str(raw / f"{imp}_generation_*.json"))
    fr_price = _merge(str(raw / f"{imp}_price_*.json"), "price")["price"]

    gb_cap = {k: max(v.values()) * 1.15 for k, v in gb_gen.items() if v}
    fr_cap = {k: max(v.values()) * 1.15 for k, v in fr_gen.items() if v}
    return gb_gen, gb_price, fr_gen, fr_price, flow, gb_cap, fr_cap


def run(raw: pathlib.Path, label: str, candidate_rule: str = "marginal_v2",
        policy: str = "unresolve", imp: str = "fr", carbon=None, year: int = 2025, max_month: int = 12) -> dict:
    """`carbon(h) -> (p_gb, p_imp)` in EUR/tCO2. Default: the December-2025 constants,
    which is the 2025 retrospective behaviour. For the charged period pass a monthly lookup."""
    links, techs_attr, pco2_attr, eta_attr = IMPORTERS[imp]
    IMP_TECHS = getattr(INP, techs_attr)
    P_CO2_IMP = getattr(INP, pco2_attr)
    ETA_IMP = getattr(INP, eta_attr)
    gb_gen, gb_p, fr_gen, fr_p, flow, gb_cap, fr_cap = load_gb_fr(raw, imp)
    if carbon is None:
        carbon = lambda h: (INP.P_CO2_GB_EUR_PER_T, P_CO2_IMP)   # noqa: E731
    fuel = daily_ttf_fuel(INP.USD_PER_EUR_2025_12)
    eta_mid = sum(ETA_IMP) / 2.0

    hours = sorted(set(gb_p) & set(fr_p) & set(flow)
                   & set(gb_gen.get("Fossil Gas", {})) & set(fr_gen.get("Fossil gas", {})))
    hours = [h for h in hours if datetime.fromtimestamp(h, tz=timezone.utc).year == year]  # A18
    # charged window is bounded by the fuel/carbon series coverage (Jan-Aug 2026)
    hours = [h for h in hours if datetime.fromtimestamp(h, tz=timezone.utc).month <= max_month]
    exp = [h for h in hours if flow[h] < 0]              # GB -> FR export = NEGATIVE

    tally, miss = Counter(), Counter()
    tot = det = mis = ali = 0.0
    det_h = agree_h = cov_h = 0
    agree_mwh = cov_mwh = 0.0
    val_ali = val_mis = 0.0
    # 🔴 A19 relabel (orientation review 2026-09-21). These are NOT "emissions caused by the
    # charge" and NOT "levy paid":
    #   contrast = (local tCO2 per MWh of suppressed import) x (hour's observed import MWh).
    #     A volume-weighted LOCAL EMISSION CONTRAST. It assumes neither that the charge reduces
    #     imports (no flow-response model exists yet) nor that the slope holds over the volume.
    #   gross_levy_proxy = monthly EUA x e_dec x MWh. The 2026 certificate price is the
    #     QUARTERLY auction average (Art. 21), and Art. 9 deductions are not applied - so this is
    #     a gross exposure proxy, not a liability.
    val_dE_mis = val_dE_ali = 0.0
    levy_mis = levy_ali = 0.0
    by_month = {}

    for h in exp:
        mwh = -flow[h]
        tot += mwh
        gas, coal = fuel(h)
        pc_gb, pc_imp = carbon(h)
        gt, ft = techs_for(INP.GB_TECHS, gas, coal), techs_for(IMP_TECHS, gas, coal)

        cg = build_candidates("GB", obs(gb_gen, gb_cap, gt, h), clearing_price=gb_p[h],
                              p_co2=pc_gb, direction=Direction.REDUCE,
                              price_tolerance=INP.price_tolerance(gb_p[h]),
                              candidate_rule=candidate_rule)
        cf = build_candidates(imp.upper(), obs(fr_gen, fr_cap, ft, h), clearing_price=fr_p[h],
                              p_co2=pc_imp, direction=Direction.INCREASE,
                              price_tolerance=INP.price_tolerance(fr_p[h]),
                              candidate_rule=candidate_rule)
        sigma, reason = sign_from_sets(cg, cf, ETA_IMP,
                                       intertemporal_policy=policy)

        m = datetime.fromtimestamp(h, tz=timezone.utc).month
        bm = by_month.setdefault(m, [0.0, 0.0])
        bm[0] += mwh

        if sigma is None:
            tally["UNDETERMINED" if reason == "emission intervals overlap"
                  else "UNRESOLVED"] += 1
            continue

        det += mwh
        det_h += 1
        bm[1] += mwh
        if sigma > 0:
            mis += mwh
            tally["DETERMINED-MISDIRECTED"] += 1
        else:
            ali += mwh
            tally["DETERMINED-ALIGNED"] += 1

        ux = units_from(gt, {k: v[h] for k, v in gb_gen.items() if h in v}, gb_cap,
                        pc_gb, clearing_price=gb_p[h])
        um = units_from(ft, {k: v[h] for k, v in fr_gen.items() if h in v}, fr_cap,
                        pc_imp, clearing_price=fr_p[h])
        res = reclear_hour(ux, um, flow_mw=mwh, eta=eta_mid, cap_flow=1e6)
        chk = check_against_mefband(res, sigma, cg, cf)
        if chk["solved"]:
            if chk["agreement"]:
                agree_h += 1
                agree_mwh += mwh
            if chk["agreement"] and chk["coverage_ok"]:
                # reclear.delta_emissions is tCO2 for ONE MWh less delivered, i.e. the
                # charge's marginal effect per MWh; scale by the hour's MWh.
                dE = res.delta_emissions * mwh
                cp = cert_price(h, pc_imp)
                levy = (cp * E_DEC_GB * mwh) if cp is not None else float("nan")
                if sigma > 0:
                    val_mis += mwh
                    val_dE_mis += dE
                    levy_mis += levy
                else:
                    val_ali += mwh
                    val_dE_ali += dE
                    levy_ali += levy
            if not chk["coverage_ok"]:
                cov_h += 1
                cov_mwh += mwh
                for x in chk["responders_outside_candidate_set"]:
                    miss[x] += 1

    return {
        "label": label, "border": f"GB->{imp.upper()}", "year": year,
        "candidate_rule": candidate_rule, "intertemporal_policy": policy,
        "conditioned_on": "GB->FR export hours, all inputs present; NOT regime R1",
        "export_hours": len(exp), "hours_total": len(hours),
        "mwh_total": tot, "mwh_determined": det,
        "mwh_misdirected": mis, "mwh_aligned": ali,
        "determinacy_pct": 100 * det / tot if tot else None,
        "labels": dict(tally),
        "validation": {
            "determined_hours": det_h,
            "agreement_pct": 100 * agree_h / det_h if det_h else None,
            "agreement_volume_pct": 100 * agree_mwh / det if det else None,
            "coverage_failure_pct": 100 * cov_h / det_h if det_h else None,
            "coverage_failure_volume_pct": 100 * cov_mwh / det if det else None,
            "responders_outside": dict(miss),
            "validated_mwh_aligned": val_ali,
            "validated_mwh_misdirected": val_mis,
            "local_emission_contrast_tco2_misdirected": val_dE_mis,
            "local_emission_contrast_tco2_aligned": val_dE_ali,
            "gross_levy_exposure_proxy_eur_misdirected": levy_mis,
            "gross_levy_exposure_proxy_eur_aligned": levy_ali,
            "validated_determinacy_pct": 100 * (val_ali + val_mis) / tot if tot else None,
        },
        "determinacy_by_month_pct": {m: (100 * v[1] / v[0] if v[0] else None)
                                     for m, v in sorted(by_month.items())},
    }


if __name__ == "__main__":
    r = run(pathlib.Path("data/raw/gb_fr_2025"), "GB->FR 2025, daily TTF")
    print(f"\n######## {r['label']} ########")
    print(f"  GB->FR export hours {r['export_hours']} of {r['hours_total']}, "
          f"{r['mwh_total']:,.0f} MWh")
    if r["determinacy_pct"] is not None:
        print(f"  determinacy        = {r['determinacy_pct']:.2f} %  ({r['mwh_determined']:,.0f} MWh)")
        print(f"    aligned  (s=-1)  = {r['mwh_aligned']:,.0f} MWh")
        print(f"    misdir.  (s=+1)  = {r['mwh_misdirected']:,.0f} MWh")
    v = r["validation"]
    if v["determined_hours"]:
        print(f"  AGREEMENT          = {v['agreement_pct']:.2f} %  (vol {v['agreement_volume_pct']:.2f} %)")
        print(f"  COVERAGE FAILURE   = {v['coverage_failure_pct']:.2f} %  (vol {v['coverage_failure_volume_pct']:.2f} %)")
        print(f"  outside-set        = {v['responders_outside']}")
    print("  labels:", r["labels"])
    print("  determinacy by month:",
          {m: (round(x, 1) if x is not None else None)
           for m, x in r["determinacy_by_month_pct"].items()})
    out = pathlib.Path("data/processed/year_gb_fr_2025_dailyfuel.json")
    out.write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(f"\nwritten -> {out}")
