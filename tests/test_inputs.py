"""Import-time and invariant checks on the frozen input files.

Added 2026-09-21 after a malformed edit made both input files unparseable while the
whole test suite still passed - because no test imported them. A test suite that cannot
see the inputs cannot protect the results.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from wedge import inputs_gb_fr_2025_12 as GBFR  # noqa: E402
from wedge import inputs_rs_hu_2025_12 as RSHU  # noqa: E402

ALL = {**{f"RS:{k}": v for k, v in RSHU.RS_TECHS.items()},
       **{f"HU:{k}": v for k, v in RSHU.HU_TECHS.items()},
       **{f"GB:{k}": v for k, v in GBFR.GB_TECHS.items()},
       **{f"FR:{k}": v for k, v in GBFR.FR_TECHS.items()},
       **{f"NL:{k}": v for k, v in GBFR.NL_TECHS.items()},
       **{f"BE:{k}": v for k, v in GBFR.BE_TECHS.items()}}


@pytest.mark.parametrize("key", sorted(ALL))
def test_every_tech_is_well_formed(key):
    t = ALL[key]
    lo, hi = t.ef_el_range
    assert 0.0 <= lo <= hi, f"{key}: EF range inverted or negative"
    elo, ehi = t.eta_range
    assert 0.0 < elo <= ehi <= 1.0, f"{key}: efficiency outside (0,1]"


@pytest.mark.parametrize("key", sorted(ALL))
def test_variable_renewables_are_zero_emission_and_flagged(key):
    """A10: wind, solar and run-of-river are resource-limited AND zero direct emission."""
    t = ALL[key]
    is_vre = any(s in t.name for s in ("wind", "solar", "hydro_ror"))
    assert t.variable == is_vre, f"{key}: variable flag disagrees with technology"
    if t.variable:
        assert t.ef_el_range == (0.0, 0.0)


def test_no_nuclear_cost_band_is_invented():
    """A8: nuclear SRMC is not constructible from public fuel prices."""
    for key, t in ALL.items():
        if "nuclear" in t.name:
            assert t.cost_unidentified, f"{key}: nuclear must be cost-unidentified"
            assert t.fuel_th is None or t.cost_unidentified


def test_serbian_lignite_is_not_priced_off_seaborne_coal():
    """Mine-mouth lignite has no published marginal fuel cost."""
    assert RSHU.RS_TECHS["Fossil brown coal / lignite"].cost_unidentified


def test_carbon_prices_are_jurisdiction_specific():
    """the study design: never assume EUA applies everywhere."""
    assert RSHU.P_CO2_RS_EUR_PER_T == 0.0          # Serbia, 2025
    assert GBFR.P_CO2_GB_EUR_PER_T != GBFR.P_CO2_FR_EUR_PER_T   # UKA != EUA, no linkage


def test_loss_factor_intervals_are_valid():
    for eta in (RSHU.ETA_LINK_RANGE, GBFR.ETA_LINK_RANGE):
        assert 0.0 < eta[0] <= eta[1] <= 1.0
