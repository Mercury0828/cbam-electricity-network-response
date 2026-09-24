"""Adversarial tests for `mefband`.

Guide is explicit that a clean result on random input is not evidence. Every case here
is constructed to break something specific, and each names the guide clause it defends.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from wedge.mefband import (  # noqa: E402
    CandidateSet, Direction, Observation, Tech,
    build_candidates, can_move, covers, sign_from_sets, srmc_band,
)

# ---------------------------------------------------------------- fixtures

CCGT = Tech(name="ccgt", eta_range=(0.50, 0.60), ef_el_range=(0.33, 0.40),
            vom_range=(1.0, 3.0), fuel_th=30.0)
LIGNITE = Tech(name="lignite", eta_range=(0.30, 0.38), ef_el_range=(0.95, 1.20),
               vom_range=(2.0, 5.0), fuel_th=8.0)
NUCLEAR = Tech(name="nuclear", eta_range=(0.33, 0.35), ef_el_range=(0.0, 0.0),
               vom_range=(5.0, 9.0), fuel_th=2.0)
HYDRO_RES = Tech(name="hydro_reservoir", eta_range=(1.0, 1.0), ef_el_range=(0.0, 0.0),
                 cost_unidentified=True, intertemporal=True)
WIND = Tech(name="wind", eta_range=(1.0, 1.0), ef_el_range=(0.0, 0.0),
            vom_range=(0.0, 1.0), fuel_th=0.0)


def obs(tech, output, cap, tmin=0.0, interpolated=False):
    return Observation(tech=tech, output_mw=output, available_capacity_mw=cap,
                       technical_min_mw=tmin, interpolated=interpolated)


# ---------------------------------------------------------------- unit basis

def test_carbon_term_uses_electrical_basis_exactly_once():
    """🔴 the study design step 1: the worked counterexample.

    With EF_el = 0.4, eta = 0.5 and p_CO2 = 100 the carbon component is 40 EUR/MWh, not 80.
    Dividing EF_el by eta a second time is the unit error the guide says must be tested.
    """
    t = Tech(name="t", eta_range=(0.5, 0.5), ef_el_range=(0.4, 0.4),
             vom_range=(0.0, 0.0), fuel_th=0.0)
    lo, hi = srmc_band(t, p_co2=100.0)
    assert lo == pytest.approx(40.0)
    assert hi == pytest.approx(40.0)
    assert lo != pytest.approx(80.0), "EF_el was divided by eta a second time"


def test_ef_th_round_trip():
    """EF_el = EF_th / eta, so EF_th = EF_el * eta."""
    t = Tech(name="t", eta_range=(0.4, 0.4), ef_el_range=(0.8, 0.8))
    assert t.ef_th_range == pytest.approx((0.32, 0.32))


def test_fuel_term_pairs_low_cost_with_high_efficiency():
    t = Tech(name="t", eta_range=(0.40, 0.60), ef_el_range=(0.0, 0.0), fuel_th=60.0)
    lo, hi = srmc_band(t, p_co2=0.0)
    assert lo == pytest.approx(100.0)   # 60 / 0.60
    assert hi == pytest.approx(150.0)   # 60 / 0.40


# ---------------------------------------------- directional filter (step 2)

def test_unit_at_full_capacity_can_still_reduce():
    """🔴 the study design step 2: the guide originally got this wrong."""
    o = obs(CCGT, output=100.0, cap=100.0, tmin=20.0)
    assert can_move(o, Direction.REDUCE) is True
    assert can_move(o, Direction.INCREASE) is False


def test_unit_at_technical_minimum_can_still_increase():
    o = obs(CCGT, output=20.0, cap=100.0, tmin=20.0)
    assert can_move(o, Direction.INCREASE) is True
    assert can_move(o, Direction.REDUCE) is False


def test_the_two_sides_require_opposite_directions():
    """Requiring both directions of one unit would discard admissible responders."""
    full = obs(LIGNITE, output=500.0, cap=500.0, tmin=200.0)
    assert can_move(full, Direction.REDUCE) and not can_move(full, Direction.INCREASE)


def test_interpolated_observations_are_not_evidence():
    """the study design rule 3."""
    cs = build_candidates("x", [obs(CCGT, 50, 100, 10, interpolated=True)],
                          clearing_price=70.0, p_co2=80.0,
                          direction=Direction.REDUCE, price_tolerance=5.0)
    assert cs.empty
    assert cs.reason == "no-adjustment-room"


# ------------------------------------------------------- candidate building

def test_empty_candidate_set_is_reported_not_patched():
    """the study design step 4: an empty set means model and data disagree."""
    cs = build_candidates("x", [obs(CCGT, 50, 100, 10)],
                          clearing_price=999.0, p_co2=80.0,
                          direction=Direction.REDUCE, price_tolerance=1.0)
    assert cs.empty
    assert cs.reason == "empty-candidate-set"
    assert "ccgt" in cs.price_excluded


def test_opportunity_cost_bidder_widens_rather_than_narrows():
    """the study design: unknown hydro must make us LESS certain, not more."""
    cs = build_candidates("x", [obs(HYDRO_RES, 50, 100, 0)],
                          clearing_price=45.0, p_co2=80.0,
                          direction=Direction.REDUCE, price_tolerance=5.0)
    assert not cs.empty
    assert "hydro_reservoir" in cs.unidentified_cost


# ------------------------------------------------------- the determinacy rule

def test_loss_factor_flips_the_sign_near_the_boundary():
    """🔴 the study design's worked example: eta=0.90, e_out=0.40, e_in=0.42.

    Lossless arithmetic says the charge raises emissions (0.42 > 0.40).
    Carrying eta says it lowers them (0.42 < 0.444).
    """
    exporter = CandidateSet("x", Direction.REDUCE,
                            members=[Tech("e_out", (1, 1), (0.40, 0.40))])
    importer = CandidateSet("m", Direction.INCREASE,
                            members=[Tech("e_in", (1, 1), (0.42, 0.42))])

    sigma_lossless, _ = sign_from_sets(exporter, importer, eta_link=1.0)
    assert sigma_lossless == +1, "without losses the importer looks dirtier"

    sigma_real, _ = sign_from_sets(exporter, importer, eta_link=0.90)
    assert sigma_real == -1, "carrying eta reverses the sign"


def test_overlapping_intervals_give_undetermined():
    exporter = CandidateSet("x", Direction.REDUCE,
                            members=[CCGT, LIGNITE])
    importer = CandidateSet("m", Direction.INCREASE, members=[CCGT])
    sigma, reason = sign_from_sets(exporter, importer, eta_link=1.0)
    assert sigma is None
    assert reason == "emission intervals overlap"


def test_ranges_not_point_values_prevent_false_separation():
    """🔴 the study design point 2.

    Two technologies whose POINT factors are 0.40 and 0.45 look separated. Carrying their
    admissible ranges, which overlap, correctly refuses to decide.
    """
    narrow_x = CandidateSet("x", Direction.REDUCE,
                            members=[Tech("a", (1, 1), (0.40, 0.40))])
    narrow_m = CandidateSet("m", Direction.INCREASE,
                            members=[Tech("b", (1, 1), (0.45, 0.45))])
    assert sign_from_sets(narrow_x, narrow_m, 1.0)[0] == +1

    wide_x = CandidateSet("x", Direction.REDUCE,
                          members=[Tech("a", (1, 1), (0.35, 0.48))])
    wide_m = CandidateSet("m", Direction.INCREASE,
                          members=[Tech("b", (1, 1), (0.41, 0.52))])
    assert sign_from_sets(wide_x, wide_m, 1.0)[0] is None


def test_intertemporal_candidate_is_unresolved_not_zero():
    """the study design: reservoir hydro at the margin moves energy between intervals."""
    x = CandidateSet("x", Direction.REDUCE, members=[HYDRO_RES], has_intertemporal=True)
    m = CandidateSet("m", Direction.INCREASE, members=[CCGT])
    sigma, reason = sign_from_sets(x, m, 0.95)
    assert sigma is None
    assert reason == "intertemporal-adjustment-plausible"


def test_empty_set_never_yields_a_sign():
    x = CandidateSet("x", Direction.REDUCE, members=[])
    m = CandidateSet("m", Direction.INCREASE, members=[CCGT])
    assert sign_from_sets(x, m, 1.0)[0] is None


@pytest.mark.parametrize("eta", [0.0, -0.1, 1.5])
def test_invalid_eta_rejected(eta):
    x = CandidateSet("x", Direction.REDUCE, members=[CCGT])
    m = CandidateSet("m", Direction.INCREASE, members=[LIGNITE])
    with pytest.raises(ValueError):
        sign_from_sets(x, m, eta)


# ------------------------------------------------------------- 🔴 COVERAGE

def test_high_determinacy_can_coexist_with_coverage_failure():
    """🔴 THE K5b CASE. This is the single most important test in the project.

    A too-narrow price tolerance filters the TRUE responder out of the exporter set. What
    survives is clean nuclear, so the rule reports a confident sign with a perfect
    determinacy rate — and the sign is wrong.

    Ground truth: lignite was on the margin in the exporter zone, so the importer (CCGT,
    ~0.36) is CLEANER than the exporter (lignite ~1.07), and suppressing the import would
    RAISE emissions. True sigma = +1... and the filtered machinery confidently returns -1.
    """
    exporter_obs = [
        obs(NUCLEAR, output=800.0, cap=800.0, tmin=700.0),
        obs(LIGNITE, output=500.0, cap=600.0, tmin=200.0),   # the TRUE responder
    ]
    # clearing price sits in nuclear's band but a tight tolerance excludes lignite
    cs = build_candidates("x", exporter_obs, clearing_price=12.0, p_co2=5.0,
                          direction=Direction.REDUCE, price_tolerance=0.5)

    assert not cs.empty
    assert covers(cs, "lignite") is False, "the true responder was filtered out"

    importer = CandidateSet("m", Direction.INCREASE, members=[CCGT])
    sigma, _ = sign_from_sets(cs, importer, eta_link=0.95)

    # determinacy looks excellent: a sign was produced
    assert sigma is not None, "the narrow filter produced a CONFIDENT sign"
    # and it is the opposite of the truth
    assert sigma == +1 or sigma == -1
    truth_x = CandidateSet("x", Direction.REDUCE, members=[LIGNITE])
    true_sigma, _ = sign_from_sets(truth_x, importer, eta_link=0.95)
    assert true_sigma == -1, "with the true responder, charging helps"
    assert sigma != true_sigma, (
        "coverage failure produced a determinate sign opposite to the truth - "
        "exactly the K5b failure mode, and invisible in the determinacy rate"
    )


def test_coverage_holds_when_tolerance_admits_the_true_responder():
    exporter_obs = [
        obs(NUCLEAR, output=800.0, cap=800.0, tmin=700.0),
        obs(LIGNITE, output=500.0, cap=600.0, tmin=200.0),
    ]
    cs = build_candidates("x", exporter_obs, clearing_price=25.0, p_co2=5.0,
                          direction=Direction.REDUCE, price_tolerance=15.0)
    assert covers(cs, "lignite") is True
    importer = CandidateSet("m", Direction.INCREASE, members=[CCGT])
    sigma, reason = sign_from_sets(cs, importer, eta_link=0.95)
    # with both nuclear and lignite admissible the interval is wide and we refuse to decide
    assert sigma is None
    assert reason == "emission intervals overlap"


def test_negative_price_hour_keeps_wind_admissible():
    """the study design step 1: support schemes can make wind marginal at negative prices."""
    cs = build_candidates("x", [obs(WIND, 300.0, 400.0, 0.0)],
                          clearing_price=-5.0, p_co2=80.0,
                          direction=Direction.REDUCE, price_tolerance=6.0)
    assert covers(cs, "wind")


# ------------------------------------------------- interval loss factor

def test_interval_eta_requires_the_sign_to_hold_for_every_admissible_eta():
    """No RS->HU loss factor is published, so eta is carried as an interval.

    A sign is declared only if it survives the whole interval. Here the exporter is at
    0.40 and the importer at 0.42: with eta = 1.0 the importer looks dirtier (+1), with
    eta = 0.90 it does not. Over the interval [0.90, 1.0] nothing is determined.
    """
    x = CandidateSet("x", Direction.REDUCE, members=[Tech("e_out", (1, 1), (0.40, 0.40))])
    m = CandidateSet("m", Direction.INCREASE, members=[Tech("e_in", (1, 1), (0.42, 0.42))])

    assert sign_from_sets(x, m, 1.0)[0] == +1
    assert sign_from_sets(x, m, 0.90)[0] == -1
    assert sign_from_sets(x, m, (0.90, 1.0))[0] is None, "must not decide across the interval"


def test_interval_eta_still_decides_when_separation_is_wide():
    x = CandidateSet("x", Direction.REDUCE, members=[Tech("lig", (1, 1), (1.00, 1.15))])
    m = CandidateSet("m", Direction.INCREASE, members=[Tech("ccgt", (1, 1), (0.33, 0.37))])
    sigma, _ = sign_from_sets(x, m, (0.9669, 1.0))
    assert sigma == -1, "lignite exporter into a gas importer is aligned under any eta here"


def test_interval_eta_rejects_inverted_bounds():
    x = CandidateSet("x", Direction.REDUCE, members=[CCGT])
    m = CandidateSet("m", Direction.INCREASE, members=[LIGNITE])
    with pytest.raises(ValueError):
        sign_from_sets(x, m, (1.0, 0.9))


# ============================================================================
# 🔴 THE STRUCTURAL RESULT FOUND IN THE SCREENING PHASE.
# Discovered empirically (0.00% determinacy on BOTH RS->HU and GB->FR,
# December 2025) and then proved here.
# ============================================================================

def test_a_zero_ef_candidate_on_the_importer_side_makes_plus_one_impossible():
    """sigma = +1 requires min EF(C_m) > max EF(C_x)/eta.

    If the IMPORTER set contains any zero-emission technology then min EF(C_m) = 0, and
    since max EF(C_x)/eta >= 0 the inequality can never hold. No amount of separation on
    the exporter side can rescue it.
    """
    x = CandidateSet("x", Direction.REDUCE, members=[Tech("lig", (1, 1), (1.0, 1.2))])
    m = CandidateSet("m", Direction.INCREASE, members=[
        Tech("ccgt", (1, 1), (0.33, 0.37)),
        Tech("hydro", (1, 1), (0.0, 0.0)),      # one zero-EF candidate is enough
    ])
    assert sign_from_sets(x, m, 1.0)[0] != +1


def test_a_zero_ef_candidate_on_the_exporter_side_makes_minus_one_impossible():
    """sigma = -1 requires max EF(C_m) < min EF(C_x)/eta.

    If the EXPORTER set contains any zero-emission technology then min EF(C_x) = 0, so the
    requirement becomes max EF(C_m) < 0, which is impossible.
    """
    x = CandidateSet("x", Direction.REDUCE, members=[
        Tech("lig", (1, 1), (1.0, 1.2)),
        Tech("hydro", (1, 1), (0.0, 0.0)),
    ])
    m = CandidateSet("m", Direction.INCREASE, members=[Tech("ccgt", (1, 1), (0.33, 0.37))])
    assert sign_from_sets(x, m, 1.0)[0] != -1


def test_zero_ef_on_both_sides_makes_determinacy_structurally_impossible():
    """🔴 The finding. Both sides carrying a zero-EF candidate => sigma is ALWAYS undecided.

    This is not a property of any particular border or month. Every European bidding zone
    has hydro, wind, solar, nuclear or biomass admissible in most hours, and the study design
    requires opportunity-cost bidders to be ADMITTED with a ~zero direct operating factor
    and an unidentified cost band - so they are never removed by the price filter.

    The consequence is that the the study design determinacy rule, applied literally, can only
    ever return a determined sign in the rare hours where at least one side has NO
    zero-emission candidate at all.
    """
    for eta in (1.0, 0.95, (0.9669, 1.0)):
        x = CandidateSet("x", Direction.REDUCE, members=[
            Tech("lig", (1, 1), (1.00, 1.20)), Tech("hydro_x", (1, 1), (0.0, 0.0))])
        m = CandidateSet("m", Direction.INCREASE, members=[
            Tech("ccgt", (1, 1), (0.33, 0.37)), Tech("hydro_m", (1, 1), (0.0, 0.0))])
        sigma, reason = sign_from_sets(x, m, eta)
        assert sigma is None, f"eta={eta} unexpectedly produced a sign"
        assert reason == "emission intervals overlap"


def test_determinacy_returns_as_soon_as_one_side_has_no_zero_ef_candidate():
    """The converse, which is what a second border-month must achieve."""
    x = CandidateSet("x", Direction.REDUCE, members=[Tech("lig", (1, 1), (1.00, 1.20))])
    m = CandidateSet("m", Direction.INCREASE, members=[
        Tech("ccgt", (1, 1), (0.33, 0.37)), Tech("hydro_m", (1, 1), (0.0, 0.0))])
    sigma, _ = sign_from_sets(x, m, (0.9669, 1.0))
    assert sigma == -1, "exporter with no zero-EF candidate is decidable as aligned"


# ============================================================================
# The revised candidate rule `marginal_v2` (amendment A7, decision D-013).
# Design choice (2026-09-21): candidates must be PLAUSIBLY AT THE
# MARGIN, not merely able to move.
# ============================================================================

from wedge.mefband import is_partially_loaded  # noqa: E402


def test_partially_loaded_rejects_units_pinned_at_a_bound():
    """A unit at available capacity is infra-marginal; at technical minimum, extra-marginal.
    Only a partially loaded unit can set the price (Blume-Werry et al. 2021; JRC JRC134300).
    """
    assert is_partially_loaded(obs(CCGT, output=50, cap=100, tmin=10)) is True
    assert is_partially_loaded(obs(CCGT, output=100, cap=100, tmin=10)) is False
    assert is_partially_loaded(obs(CCGT, output=10, cap=100, tmin=10)) is False


def test_marginal_v2_suppresses_unidentified_cost_when_the_merit_order_explains_the_price():
    """Clause (iii): cost-unidentified technologies are a FALLBACK, not a default.

    This is what breaks the B16 deadlock - it stops a zero-EF opportunity-cost bidder from
    being admitted in every hour regardless of price.
    """
    observations = [obs(CCGT, 50, 100, 10), obs(HYDRO_RES, 50, 100, 0)]
    price = 85.0   # inside CCGT's band [77.4, 95.0] at p_co2=80

    v1 = build_candidates("z", observations, clearing_price=price, p_co2=80.0,
                          direction=Direction.REDUCE, price_tolerance=5.0,
                          candidate_rule="guide_v1")
    assert {t.name for t in v1.members} == {"ccgt", "hydro_reservoir"}

    v2 = build_candidates("z", observations, clearing_price=price, p_co2=80.0,
                          direction=Direction.REDUCE, price_tolerance=5.0,
                          candidate_rule="marginal_v2")
    assert {t.name for t in v2.members} == {"ccgt"}
    assert "hydro_reservoir" in v2.fallback_suppressed


def test_marginal_v2_falls_back_when_no_identified_technology_explains_the_price():
    """If the merit order does NOT explain the price we admit that we do not know."""
    observations = [obs(CCGT, 50, 100, 10), obs(HYDRO_RES, 50, 100, 0)]
    v2 = build_candidates("z", observations, clearing_price=500.0, p_co2=80.0,
                          direction=Direction.REDUCE, price_tolerance=5.0,
                          candidate_rule="marginal_v2")
    assert {t.name for t in v2.members} == {"hydro_reservoir"}
    assert "ccgt" in v2.price_excluded


def test_marginal_v2_reintroduces_the_k5b_coverage_risk_and_records_it():
    """🔴 The cost of clause (iii), recorded rather than hidden.

    Suppressing the fallback can exclude a technology that truly was marginal. The rule must
    therefore always record what it suppressed, so the exposure is auditable and can be
    checked against the the study design re-clearing model.
    """
    observations = [obs(CCGT, 50, 100, 10), obs(HYDRO_RES, 50, 100, 0)]
    v2 = build_candidates("z", observations, clearing_price=85.0, p_co2=80.0,
                          direction=Direction.REDUCE, price_tolerance=5.0,
                          candidate_rule="marginal_v2")
    assert covers(v2, "hydro_reservoir") is False        # the suppressed one is NOT covered
    assert v2.fallback_suppressed == ["hydro_reservoir"]  # but it IS recorded


def test_guide_v1_remains_available_so_before_and_after_can_both_be_reported():
    """The study design requires before/after when a frozen study control changes."""
    observations = [obs(CCGT, 50, 100, 10), obs(HYDRO_RES, 50, 100, 0)]
    for rule in ("guide_v1", "marginal_v2"):
        cs = build_candidates("z", observations, clearing_price=85.0, p_co2=80.0,
                              direction=Direction.REDUCE, price_tolerance=5.0,
                              candidate_rule=rule)
        assert not cs.empty
    with pytest.raises(ValueError):
        build_candidates("z", observations, clearing_price=85.0, p_co2=80.0,
                         direction=Direction.REDUCE, price_tolerance=5.0,
                         candidate_rule="something_else")
