"""Frozen inputs for the first border-month: RS -> HU, December 2025.

🔴 Every number here carries its source. Nothing is estimated, interpolated or recalled.
Where a value could not be sourced it is represented as an explicit interval or marked
unidentified, never filled in with a plausible number.

🔴 **This is a RETROSPECTIVE month.** CBAM electricity charging begins 2026-01-01, so
December 2025 is the rule applied to historical market conditions. The study design permits this
explicitly and requires that retrospective and charged-period quantities are never summed.
The month was chosen because the World Bank fuel series ends 2025M12, not because of any
result.

Sources are listed in the data repository (DATA_SOURCES.md).
"""

from __future__ import annotations

from .mefband import Tech

# --------------------------------------------------------------- unit conversions
MMBTU_PER_MWH = 3.412142       # exact-ish: 1 MWh = 3.412142 MMBtu
USD_PER_EUR_2025_12 = 1.16     # 🔴 ASSUMPTION, see NOTE_FX below

# --------------------------------------------------------------- carbon prices
# Q1: KOBiZE Raport z rynku CO2 No. 165, Dec 2025, Table 2 p.3 - EUA secondary-market
# spot compiled from ICE and EEX. Arithmetic monthly average.
P_CO2_HU_EUR_PER_T = 83.90
P_CO2_HU_RANGE = (81.73, 86.32)          # monthly min / max from the same table

# Q2: Serbia had NO domestic ETS or GHG emissions tax applied to generation in calendar
# 2025. Serbian Ministry of Mining and Energy, Draft Just Transition Action Plan, May
# 2025, p.8: Serbia is outside the EU ETS and "does not have an established greenhouse gas
# emissions taxation system". Law 109/2025 was adopted in December 2025 and applies from
# 2026-01-01 at EUR 4/tCO2 (IEA policy record; World Bank carbon pricing dashboard).
P_CO2_RS_EUR_PER_T = 0.0
NOTE_RS_CARBON = (
    "Absence of an applicable charge in 2025, evidenced affirmatively. No source publishes "
    "a literal 'Serbia: EUR 0/tCO2 in 2025'. From 2026-01-01 Serbia levies EUR 4/tCO2 under "
    "Law 109/2025 - 🔴 which is an Article 9 deduction candidate for the NET charge `c` in "
    "the charged period and is handled separately."
)

# --------------------------------------------------------------- fuel prices
# World Bank Commodity Markets 'Pink Sheet', Monthly Prices, row 2025M12.
GAS_EUROPE_USD_PER_MMBTU = 9.48260496782
COAL_SA_USD_PER_TONNE = 90.88

GAS_EUR_PER_MWH_TH = (GAS_EUROPE_USD_PER_MMBTU * MMBTU_PER_MWH) / USD_PER_EUR_2025_12

NOTE_FX = (
    "🔴 The USD/EUR rate is an ASSUMPTION, not yet sourced. It scales every fuel cost "
    "linearly and therefore shifts every SRMC band. It is a pre-registered sensitivity axis "
    "and must be replaced with an ECB monthly reference rate before any reported number."
)

# 🔴 Serbian lignite is MINE-MOUTH, not internationally traded. Pricing it off API2 or
# South African coal would be simply wrong: those are seaborne hard-coal markers and the
# Serbian plants burn domestic lignite from adjacent mines. No public mine-mouth cost was
# sourced, so the cost is declared UNIDENTIFIED and the technology enters the candidate set
# without a price filter. That WIDENS the candidate set and REDUCES determinacy, which is
# the honest direction of the error.
LIGNITE_COST_UNIDENTIFIED = True
NOTE_LIGNITE = (
    "Serbian lignite SRMC is not constructible from published fuel prices. Treated as "
    "cost-unidentified, exactly as the study design treats opportunity-cost bidders. This lowers "
    "determinacy rather than raising it."
)

# --------------------------------------------------------------- emission factors
# Serbian lignite, ELECTRICAL basis, plant averages 2015-2019:
#   Nikola Tesla A 1.1134, Nikola Tesla B 1.0556, Kostolac A 1.1504, Kostolac B 1.0107
# Klimenta, Mihajlovic, Ristic & Andriukaitis (2022), Energies 15(13):4792, Table A2.
RS_LIGNITE_EF_EL = (1.0107, 1.1504)      # tCO2 / MWh electrical
RS_LIGNITE_ETA = (0.309, 0.373)          # net electrical efficiency; NT A low, Kostolac B3 design

# IPCC 2006 GL Vol.2 Ch.2 Table 2.2, THERMAL-input basis, converted to tCO2/MWh_th:
IPCC_GAS_EF_TH = (0.19548, 0.20988)      # default 0.20196
IPCC_LIGNITE_EF_TH = (0.32724, 0.414)    # default 0.3636

# 🔴 A14, 2026-09-21. The MARGINAL gas unit is the least efficient one running, not the best
# CCGT. 57-59% came from two modern CCGTs (Dunamenti, Gonyu) and the source said explicitly it
# was not the fleet envelope - a misapplication that produced the HU_ccgt coverage failures.
# Replaced by a SOURCED EU-wide envelope for EXISTING gas units, net electrical, LHV:
#   OCGT >=50 MWth 33-41.5%  |  gas steam 38-40%  |  CCGT 50-600 MWth 46-54%
#   CCGT >=600 MWth 50-60%   |  gas engines 35-44%
# Commission Implementing Decision (EU) 2021/2326, LCP BAT conclusions, BAT 40, Table 23.
# Consistent with the Hungarian OCGT at Ajka/Bakony, 40% net (Hungarian Energy Office 2012).
GAS_FLEET_ETA = (0.33, 0.60)
HU_CCGT_ETA = GAS_FLEET_ETA
# EF_el = EF_th / eta. Converted ONCE, here, and never divided by eta again downstream.
HU_CCGT_EF_EL = (IPCC_GAS_EF_TH[0] / HU_CCGT_ETA[1],
                 IPCC_GAS_EF_TH[1] / HU_CCGT_ETA[0])        # ~ (0.3313, 0.3682)

# Hungarian lignite (Matra): net ELECTRICAL efficiency NOT VERIFIED. The operator's
# published 31.7% mixes heat and electricity output. A wide efficiency interval is used and
# flagged, which widens the band and lowers determinacy.
HU_LIGNITE_ETA = (0.28, 0.36)            # 🔴 NOT SOURCED - declared wide, sensitivity axis
HU_LIGNITE_EF_EL = (IPCC_LIGNITE_EF_TH[0] / HU_LIGNITE_ETA[1],
                    IPCC_LIGNITE_EF_TH[1] / HU_LIGNITE_ETA[0])   # ~ (0.909, 1.479)

# --------------------------------------------------------------- loss factor
# 🔴 NO border-specific RS->HU loss factor is published, and none was found.
# National transmission losses 2024: EMS (Serbia) 2.08%, MAVIR (Hungary) 1.23%.
# eta_l is therefore carried as an INTERVAL rather than invented as a point value.
# Interval arithmetic is applied in the determinacy rule: the test for sigma=+1 uses the
# eta that makes it HARDEST (the minimum), and vice versa.
ETA_LINK_RANGE = (1.0 - 0.0208 - 0.0123, 1.0)    # ~ (0.9669, 1.0)
NOTE_ETA = (
    "Not border-specific. The lower bound stacks both national transmission loss rates as a "
    "conservative worst case; the upper bound is lossless. The study design requires eta to be "
    "carried everywhere or the hour left unresolved - carrying it as an interval satisfies "
    "that and is honest about what is not known."
)

# --------------------------------------------------------------- technology sets
VOM_THERMAL = (1.0, 4.0)      # EUR/MWh el, declared band; sensitivity axis

RS_TECHS = {
    "Fossil brown coal / lignite": Tech(
        name="RS_lignite", eta_range=RS_LIGNITE_ETA, ef_el_range=RS_LIGNITE_EF_EL,
        vom_range=VOM_THERMAL, fuel_th=None, cost_unidentified=LIGNITE_COST_UNIDENTIFIED),
    "Fossil gas": Tech(
        name="RS_gas", eta_range=GAS_FLEET_ETA,
        ef_el_range=(IPCC_GAS_EF_TH[0] / GAS_FLEET_ETA[1], IPCC_GAS_EF_TH[1] / GAS_FLEET_ETA[0]),
        vom_range=VOM_THERMAL, fuel_th=GAS_EUR_PER_MWH_TH),
    "Hydro water reservoir": Tech(
        name="RS_hydro_res", eta_range=(1.0, 1.0), ef_el_range=(0.0, 0.0),
        cost_unidentified=True, intertemporal=True),
    "Hydro Run-of-River": Tech(
        name="RS_hydro_ror", eta_range=(1.0, 1.0), ef_el_range=(0.0, 0.0),
        vom_range=(0.0, 1.0), fuel_th=0.0, variable=True),
    "Hydro pumped storage": Tech(
        name="RS_pumped", eta_range=(1.0, 1.0), ef_el_range=(0.0, 0.0),
        cost_unidentified=True, intertemporal=True),
    "Wind onshore": Tech(
        name="RS_wind", eta_range=(1.0, 1.0), ef_el_range=(0.0, 0.0),
        vom_range=(0.0, 1.0), fuel_th=0.0, variable=True),
    "Solar": Tech(
        name="RS_solar", eta_range=(1.0, 1.0), ef_el_range=(0.0, 0.0),
        vom_range=(0.0, 1.0), fuel_th=0.0, variable=True),
    "Biomass": Tech(
        name="RS_biomass", eta_range=(0.25, 0.35), ef_el_range=(0.0, 0.0),
        cost_unidentified=True, must_run=True),
}

HU_TECHS = {
    "Fossil gas": Tech(
        name="HU_ccgt", eta_range=HU_CCGT_ETA, ef_el_range=HU_CCGT_EF_EL,
        vom_range=VOM_THERMAL, fuel_th=GAS_EUR_PER_MWH_TH),
    "Fossil brown coal / lignite": Tech(
        # A15 (a partially identified fuel>=0 band) was TRIED and REVERTED on 2026-09-21: it
        # cut validated volume from 62,074 to 7,540 MWh by opening a mefband/reclear
        # disagreement that only unit-commitment modelling could resolve, which the study design
        # forbids. Matra stays cost-unidentified.
        name="HU_lignite", eta_range=HU_LIGNITE_ETA, ef_el_range=HU_LIGNITE_EF_EL,
        vom_range=VOM_THERMAL, fuel_th=None, cost_unidentified=True),
    # 🔴 CORRECTED 2026-09-21, same reason as the GB/FR file: the previous fuel_th and VOM
    # were invented. Nuclear SRMC is not constructible from public fuel prices.
    "Nuclear": Tech(
        name="HU_nuclear", eta_range=(0.33, 0.35), ef_el_range=(0.0, 0.0),
        cost_unidentified=True, must_run=True),
    "Hydro Run-of-River": Tech(
        name="HU_hydro_ror", eta_range=(1.0, 1.0), ef_el_range=(0.0, 0.0),
        vom_range=(0.0, 1.0), fuel_th=0.0, variable=True),
    "Wind onshore": Tech(
        name="HU_wind", eta_range=(1.0, 1.0), ef_el_range=(0.0, 0.0),
        vom_range=(0.0, 1.0), fuel_th=0.0, variable=True),
    "Solar": Tech(
        name="HU_solar", eta_range=(1.0, 1.0), ef_el_range=(0.0, 0.0),
        vom_range=(0.0, 1.0), fuel_th=0.0, variable=True),
    "Biomass": Tech(
        name="HU_biomass", eta_range=(0.25, 0.35), ef_el_range=(0.0, 0.0),
        cost_unidentified=True, must_run=True),
}

# Pre-registered price tolerance, design decision D-009 section D.
def price_tolerance(clearing_price: float) -> float:
    return max(5.0, 0.10 * abs(clearing_price))
