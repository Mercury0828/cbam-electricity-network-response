"""Frozen inputs for the second border-month: GB -> FR, December 2025.

RS -> HU returned zero determined
hours because both sides are dominated by technologies with no constructible SRMC band and
Serbia had no carbon price. GB -> FR is the opposite case and tests whether the failure is
specific to that border or general to the method.

🔴 Every number carries its source. Nothing is estimated or recalled.
🔴 RETROSPECTIVE month, as for RS -> HU.
"""

from __future__ import annotations

from .mefband import Tech

MMBTU_PER_MWH = 3.412142
KGTJ_TO_TCO2_PER_MWH_TH = 1.0 / 277777.7778     # 1 TJ = 277.7778 MWh; kg -> t is /1000

# --------------------------------------------------------------- carbon prices
# GB: UK ETS. DESNZ Cost Containment Mechanism full table, "Month 6 average price" for the
# Dec-2025 observation = GBP 62.13/tCO2. Series: UKA FUTURES daily settlement, averaged over
# published trading days (DESNZ uses the December 2026 contract for Dec-2025 observations).
UKA_GBP_PER_T = 62.13
# Bank of England series XUMAERS (via ONS series THAP), December 2025 monthly average.
GBP_PER_EUR_2025_12 = 1.1429
P_UKA_GB_EUR_PER_T = UKA_GBP_PER_T * GBP_PER_EUR_2025_12        # ~ 71.01 (UKA only; credit scenario)
# 🔴 A37: GB generators also pay Carbon Price Support (CPS, GBP 18/tCO2). Dispatch carbon cost =
# UKA + CPS. See wedge/carbon.py.
from wedge.carbon import CPS_GBP_PER_T as _CPS, CPS_ON as _CPS_ON   # noqa: E402
P_CO2_GB_EUR_PER_T = (UKA_GBP_PER_T + (_CPS if _CPS_ON else 0.0)) * GBP_PER_EUR_2025_12   # ~ 91.58

# 🔴 No UK-EU ETS linkage was in force in December 2025 (EC-UK joint statement,
# 17 December 2025). So the EU allowance price does NOT apply to GB generators. This is
# exactly the trap to avoid: "never assume EUA applies everywhere".
NOTE_NO_LINKAGE = (
    "UK ETS and EU ETS were NOT linked in Dec 2025. GB generators face UKA, FR generators "
    "face EUA. Using EUA on both sides would mis-order the merit stacks."
)

# FR: EU ETS. KOBiZE Raport z rynku CO2 No.165, Dec 2025, Table 2 p.3, arithmetic monthly
# average of EUA secondary-market spot compiled from ICE and EEX.
P_CO2_FR_EUR_PER_T = 83.90

# --------------------------------------------------------------- fuel prices
# World Bank Pink Sheet, Monthly Prices, 2025M12, "Natural gas, Europe".
GAS_EUROPE_USD_PER_MMBTU = 9.48260496782
USD_PER_EUR_2025_12 = 1.16      # 🔴 ASSUMPTION, same as the RS->HU month. Sensitivity axis.
GAS_EUR_PER_MWH_TH = (GAS_EUROPE_USD_PER_MMBTU * MMBTU_PER_MWH) / USD_PER_EUR_2025_12

NOTE_GB_GAS = (
    "🔴 GB gas trades at NBP, not TTF. The World Bank 'Natural gas, Europe' series is a TTF-"
    "based European marker and is used here as a PROXY for NBP. The study design requires a proxy "
    "to be named in the paper and carried in the sensitivity analysis. It is named here."
)

# Coal: World Bank 'Coal, South African' 2025M12 = 90.88 USD/tonne.
# Net calorific value of South African export coal is conventionally ~6000 kcal/kg NAR,
# i.e. 6.978 MWh_th per tonne. 🔴 The 6000 kcal/kg figure is a MARKET CONVENTION for the
# API4 grade, not a sourced measurement of any particular cargo.
COAL_SA_USD_PER_TONNE = 90.88
COAL_MWH_TH_PER_TONNE = 6.978
COAL_EUR_PER_MWH_TH = (COAL_SA_USD_PER_TONNE / COAL_MWH_TH_PER_TONNE) / USD_PER_EUR_2025_12

# --------------------------------------------------------------- IPCC factors
# IPCC 2006 Guidelines Vol.2 Ch.2 Table 2.2, read directly from the IPCC PDF (p.16 of the
# chapter file). Units as published: kg CO2 / TJ on a NET calorific basis.
#                       default   lower    upper
IPCC = {
    "natural_gas":      (56100,   54300,   58300),
    "other_bituminous": (94600,   89500,   99700),
    "gas_diesel_oil":   (74100,   72600,   74800),
    "residual_fuel_oil":(77400,   75500,   78800),
}


def _th(key: str) -> tuple[float, float]:
    """(lower, upper) thermal-basis factor in tCO2 per MWh thermal input."""
    _, lo, hi = IPCC[key]
    return (lo * KGTJ_TO_TCO2_PER_MWH_TH, hi * KGTJ_TO_TCO2_PER_MWH_TH)


GAS_TH = _th("natural_gas")            # ~ (0.19548, 0.20988)
COAL_TH = _th("other_bituminous")      # ~ (0.32220, 0.35892)
OIL_TH = _th("residual_fuel_oil")      # ~ (0.27180, 0.28368)

# --------------------------------------------------------------- efficiencies
# 🔴 NOT plant-specific sourced values. Declared intervals, wide enough to be conservative,
# and carried as a pre-registered sensitivity axis.
# 🔴 A14: sourced EU envelope for EXISTING gas units, OCGT 33% to large CCGT 60% net LHV,
# Commission Implementing Decision (EU) 2021/2326, BAT 40, Table 23. Consistent with GB's
# Peterborough OCGT at 32.05% net, ISO-corrected (Environment Agency permit decision, 2020).
CCGT_ETA = (0.33, 0.60)
COAL_ETA = (0.33, 0.42)
OIL_ETA = (0.30, 0.40)

VOM = (1.0, 4.0)


def _ef_el(th: tuple[float, float], eta: tuple[float, float]) -> tuple[float, float]:
    """EF_el = EF_th / eta. Converted ONCE, here. Never divided by eta again downstream."""
    return (th[0] / eta[1], th[1] / eta[0])


# --------------------------------------------------------------- loss factor
# Published PER LINK, unlike RS->HU:
#   IFA      1.17%  (National Grid / RTE loss factor notice, 13 Dec 2011)
#   IFA2     2.950% overall (National Grid)
#   ElecLink 2.9%  end-to-end (ElecLink publications)
# eta is carried as an interval spanning the three links.
ETA_LINK_RANGE = (1.0 - 0.0295, 1.0 - 0.0117)        # ~ (0.9705, 0.9883)
NOTE_ETA = (
    "Per-link published loss factors exist for GB->FR (IFA 1.17%, IFA2 2.950%, ElecLink "
    "2.9%). The interval spans the three links because the flow is not attributed per link "
    "in this run. This is materially better evidence than RS->HU, where no border-specific "
    "loss factor is published at all."
)

# --------------------------------------------------------------- technology sets
# GB, Elexon `psrType` names.
GB_TECHS = {
    "Fossil Gas": Tech("GB_ccgt", CCGT_ETA, _ef_el(GAS_TH, CCGT_ETA), VOM,
                       fuel_th=GAS_EUR_PER_MWH_TH),
    "Fossil Hard coal": Tech("GB_coal", COAL_ETA, _ef_el(COAL_TH, COAL_ETA), VOM,
                             fuel_th=COAL_EUR_PER_MWH_TH),
    "Fossil Oil": Tech("GB_oil", OIL_ETA, _ef_el(OIL_TH, OIL_ETA), VOM,
                       fuel_th=COAL_EUR_PER_MWH_TH * 2.5),   # crude proxy, flagged
    # 🔴 CORRECTED 2026-09-21. Nuclear previously carried fuel_th=2.0 and VOM (5,12), which
    # were INVENTED, not sourced - and that fabricated band produced the project's only
    # "misdirected" hour. Nuclear fuel cost is not a traded EUR/MWh_th price, so its SRMC is
    # not constructible from public fuel prices. It is therefore cost-unidentified, exactly
    # like hydro and storage. Removing an unsourced number is always correct.
    "Nuclear": Tech("GB_nuclear", (0.33, 0.35), (0.0, 0.0), cost_unidentified=True, must_run=True),
    "Biomass": Tech("GB_biomass", (0.30, 0.40), (0.0, 0.0), cost_unidentified=True, must_run=True),
    "Hydro Run-of-river and poundage": Tech("GB_hydro_ror", (1.0, 1.0), (0.0, 0.0),
                                            (0.0, 1.0), fuel_th=0.0, variable=True),
    "Hydro Pumped Storage": Tech("GB_pumped", (1.0, 1.0), (0.0, 0.0),
                                 cost_unidentified=True, intertemporal=True),
    "Wind Onshore": Tech("GB_wind_on", (1.0, 1.0), (0.0, 0.0), (0.0, 1.0), fuel_th=0.0, variable=True),
    "Wind Offshore": Tech("GB_wind_off", (1.0, 1.0), (0.0, 0.0), (0.0, 1.0), fuel_th=0.0, variable=True),
    "Solar": Tech("GB_solar", (1.0, 1.0), (0.0, 0.0), (0.0, 1.0), fuel_th=0.0, variable=True),
    "Other": Tech("GB_other", (0.30, 0.50), (0.0, 0.4), cost_unidentified=True, must_run=True),
}

# FR, Energy-Charts production-type names.
FR_TECHS = {
    "Fossil gas": Tech("FR_ccgt", CCGT_ETA, _ef_el(GAS_TH, CCGT_ETA), VOM,
                       fuel_th=GAS_EUR_PER_MWH_TH),
    "Fossil hard coal": Tech("FR_coal", COAL_ETA, _ef_el(COAL_TH, COAL_ETA), VOM,
                             fuel_th=COAL_EUR_PER_MWH_TH),
    "Fossil oil": Tech("FR_oil", OIL_ETA, _ef_el(OIL_TH, OIL_ETA), VOM,
                       fuel_th=COAL_EUR_PER_MWH_TH * 2.5),
    "Nuclear": Tech("FR_nuclear", (0.33, 0.35), (0.0, 0.0), cost_unidentified=True, must_run=True),
    "Hydro Run-of-River": Tech("FR_hydro_ror", (1.0, 1.0), (0.0, 0.0), (0.0, 1.0),
                               fuel_th=0.0, variable=True),
    "Hydro water reservoir": Tech("FR_hydro_res", (1.0, 1.0), (0.0, 0.0),
                                  cost_unidentified=True, intertemporal=True),
    "Hydro pumped storage": Tech("FR_pumped", (1.0, 1.0), (0.0, 0.0),
                                 cost_unidentified=True, intertemporal=True),
    "Battery": Tech("FR_battery", (1.0, 1.0), (0.0, 0.0),
                    cost_unidentified=True, intertemporal=True),
    "Wind onshore": Tech("FR_wind_on", (1.0, 1.0), (0.0, 0.0), (0.0, 1.0), fuel_th=0.0, variable=True),
    "Wind offshore": Tech("FR_wind_off", (1.0, 1.0), (0.0, 0.0), (0.0, 1.0), fuel_th=0.0, variable=True),
    "Solar": Tech("FR_solar", (1.0, 1.0), (0.0, 0.0), (0.0, 1.0), fuel_th=0.0, variable=True),
    "Biomass": Tech("FR_biomass", (0.30, 0.40), (0.0, 0.0), cost_unidentified=True, must_run=True),
    "Waste": Tech("FR_waste", (0.20, 0.30), (0.0, 0.4), cost_unidentified=True, must_run=True),
}


def price_tolerance(clearing_price: float) -> float:
    """Pre-registered, design decision D-009 section D. Identical to RS -> HU."""
    return max(5.0, 0.10 * abs(clearing_price))


# =============================================================================
# Additional EU importer zones for the GB borders, admitted by the frozen border selection
# selection rule (charged border + hourly data both sides + exporter generation by type).
# Same sourced templates as FR: IPCC Table 2.2 factors, A14 gas envelope, A8/A13 nuclear.
# =============================================================================

# 🔴 Loss factors for BritNed (GB-NL) and Nemolink (GB-BE) were NOT sourced. The interval of the
# three SOURCED GB-FR HVDC links (IFA 1.17 %, IFA2 2.950 %, ElecLink 2.9 %) is used as a DECLARED
# PROXY for comparable GB HVDC interconnectors. It is a named proxy and a sensitivity axis, not a
# sourced value for these two links.
ETA_LINK_RANGE_NL = ETA_LINK_RANGE
ETA_LINK_RANGE_BE = ETA_LINK_RANGE

# 🔴 NL "Others" averages ~5.5 GW (~45 % of load) in the Energy-Charts / ENTSO-E NL series: a known
# catch-all, largely unspecified CHP. Its emission factor is NOT identifiable, so it is treated as
# must-run and cost-unidentified rather than assigned a guessed gas-CHP factor.
NL_TECHS = {
    "Fossil gas": Tech("NL_ccgt", CCGT_ETA, _ef_el(GAS_TH, CCGT_ETA), VOM,
                       fuel_th=GAS_EUR_PER_MWH_TH),
    "Fossil hard coal": Tech("NL_coal", COAL_ETA, _ef_el(COAL_TH, COAL_ETA), VOM,
                             fuel_th=COAL_EUR_PER_MWH_TH),
    "Nuclear": Tech("NL_nuclear", (0.33, 0.35), (0.0, 0.0), cost_unidentified=True,
                    must_run=True),
    "Biomass": Tech("NL_biomass", (0.30, 0.40), (0.0, 0.0), cost_unidentified=True,
                    must_run=True),
    "Others": Tech("NL_other", (0.30, 0.50), (0.0, 0.64), cost_unidentified=True,
                   must_run=True),
    "Waste": Tech("NL_waste", (0.20, 0.30), (0.0, 0.4), cost_unidentified=True,
                  must_run=True),
    "Wind offshore": Tech("NL_wind_off", (1.0, 1.0), (0.0, 0.0), (0.0, 1.0), fuel_th=0.0,
                          variable=True),
    "Wind onshore": Tech("NL_wind_on", (1.0, 1.0), (0.0, 0.0), (0.0, 1.0), fuel_th=0.0,
                         variable=True),
    "Solar": Tech("NL_solar", (1.0, 1.0), (0.0, 0.0), (0.0, 1.0), fuel_th=0.0,
                  variable=True),
}

BE_TECHS = {
    "Fossil gas": Tech("BE_ccgt", CCGT_ETA, _ef_el(GAS_TH, CCGT_ETA), VOM,
                       fuel_th=GAS_EUR_PER_MWH_TH),
    "Fossil oil": Tech("BE_oil", OIL_ETA, _ef_el(OIL_TH, OIL_ETA), VOM,
                       fuel_th=COAL_EUR_PER_MWH_TH * 2.5),
    "Nuclear": Tech("BE_nuclear", (0.33, 0.35), (0.0, 0.0), cost_unidentified=True,
                    must_run=True),
    "Biomass": Tech("BE_biomass", (0.30, 0.40), (0.0, 0.0), cost_unidentified=True,
                    must_run=True),
    "Others": Tech("BE_other", (0.30, 0.50), (0.0, 0.64), cost_unidentified=True,
                   must_run=True),
    "Waste": Tech("BE_waste", (0.20, 0.30), (0.0, 0.4), cost_unidentified=True,
                  must_run=True),
    "Hydro Run-of-River": Tech("BE_hydro_ror", (1.0, 1.0), (0.0, 0.0), (0.0, 1.0),
                               fuel_th=0.0, variable=True),
    "Hydro pumped storage": Tech("BE_pumped", (1.0, 1.0), (0.0, 0.0),
                                 cost_unidentified=True, intertemporal=True),
    "Battery": Tech("BE_battery", (1.0, 1.0), (0.0, 0.0),
                    cost_unidentified=True, intertemporal=True),
    "Wind offshore": Tech("BE_wind_off", (1.0, 1.0), (0.0, 0.0), (0.0, 1.0), fuel_th=0.0,
                          variable=True),
    "Wind onshore": Tech("BE_wind_on", (1.0, 1.0), (0.0, 0.0), (0.0, 1.0), fuel_th=0.0,
                         variable=True),
    "Solar": Tech("BE_solar", (1.0, 1.0), (0.0, 0.0), (0.0, 1.0), fuel_th=0.0,
                  variable=True),
}

# Both NL and BE are EU ETS: EUA applies (KOBiZE Dec-2025 monthly average, as for FR).
P_CO2_NL_EUR_PER_T = P_CO2_FR_EUR_PER_T
P_CO2_BE_EUR_PER_T = P_CO2_FR_EUR_PER_T
