"""Tests for the re-clearing cross-check.

Adversarial: each case is built to break something specific.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from wedge.mefband import CandidateSet, Direction, Tech  # noqa: E402
from wedge.reclear import Unit, check_against_mefband, reclear_hour  # noqa: E402


def lignite(o, c):
    return Unit("lignite", srmc=25.0, ef_el=1.10, output_mw=o, capacity_mw=c)


def ccgt(o, c):
    return Unit("ccgt", srmc=80.0, ef_el=0.35, output_mw=o, capacity_mw=c)


def wind(o, c):
    return Unit("wind", srmc=0.0, ef_el=0.0, output_mw=o, capacity_mw=c)


def test_lignite_exporter_into_gas_importer_is_aligned():
    """Suppressing a lignite-backed import should LOWER emissions: sigma = -1."""
    res = reclear_hour([lignite(500, 800), wind(100, 100)],
                       [ccgt(300, 900), wind(50, 50)],
                       flow_mw=200.0, eta=0.97, cap_flow=1000.0)
    assert res.solved
    assert res.sign == -1, "lignite out, gas in => charging helps"
    assert res.exporter_responder == "lignite"
    assert res.importer_responder == "ccgt"


def test_clean_exporter_into_gas_importer_is_misdirected():
    """The mirror image: a zero-carbon exporter displaced by gas raises emissions."""
    res = reclear_hour([wind(500, 900)], [ccgt(300, 900)],
                       flow_mw=200.0, eta=0.97, cap_flow=1000.0)
    assert res.solved
    assert res.sign == +1, "wind out, gas in => charging hurts"


def test_marginal_perturbation_not_the_whole_charge():
    """the study design: perturb by the SAME quantity the headline estimand uses."""
    small = reclear_hour([lignite(500, 800)], [ccgt(300, 900)],
                         flow_mw=200.0, eta=1.0, cap_flow=1000.0, epsilon_mw=1.0)
    # 1 MWh less delivered: importer +1 at 0.35, exporter -1 at 1.10
    assert small.delta_emissions == pytest.approx(0.35 - 1.10, abs=1e-6)


def test_flow_too_small_is_reported_not_guessed():
    res = reclear_hour([lignite(500, 800)], [ccgt(300, 900)],
                       flow_mw=0.5, eta=1.0, cap_flow=1000.0)
    assert res.solved is False
    assert "too small" in res.note


def test_coverage_failure_is_detected_when_the_responder_is_outside_the_candidate_set():
    """🔴 THE K5b DETECTOR. This is the whole reason reclear exists.

    The model's exporter responder is lignite, but mefband's candidate set contains only
    wind. That is a coverage failure and must be flagged even though a sign was produced.
    """
    res = reclear_hour([lignite(500, 800), wind(100, 100)], [ccgt(300, 900)],
                       flow_mw=200.0, eta=0.97, cap_flow=1000.0)
    c_x = CandidateSet("x", Direction.REDUCE,
                       members=[Tech("wind", (1, 1), (0.0, 0.0))])
    c_m = CandidateSet("m", Direction.INCREASE,
                       members=[Tech("ccgt", (1, 1), (0.33, 0.37))])

    chk = check_against_mefband(res, sigma=-1, c_exporter=c_x, c_importer=c_m)
    assert chk["coverage_ok"] is False
    assert "exporter:lignite" in chk["responders_outside_candidate_set"]


def test_agreement_and_coverage_are_reported_separately():
    """the study design: a coverage failure is not a disagreement to be averaged away."""
    res = reclear_hour([lignite(500, 800)], [ccgt(300, 900)],
                       flow_mw=200.0, eta=0.97, cap_flow=1000.0)
    c_x = CandidateSet("x", Direction.REDUCE,
                       members=[Tech("lignite", (1, 1), (1.0, 1.2))])
    c_m = CandidateSet("m", Direction.INCREASE,
                       members=[Tech("ccgt", (1, 1), (0.33, 0.37))])
    chk = check_against_mefband(res, sigma=-1, c_exporter=c_x, c_importer=c_m)
    assert chk["agreement"] is True
    assert chk["coverage_ok"] is True
    assert set(chk) >= {"agreement", "coverage_ok"}


def test_agreement_false_when_mefband_undetermined():
    res = reclear_hour([lignite(500, 800)], [ccgt(300, 900)],
                       flow_mw=200.0, eta=0.97, cap_flow=1000.0)
    c_x = CandidateSet("x", Direction.REDUCE, members=[Tech("lignite", (1, 1), (1.0, 1.2))])
    c_m = CandidateSet("m", Direction.INCREASE, members=[Tech("ccgt", (1, 1), (0.33, 0.37))])
    chk = check_against_mefband(res, sigma=None, c_exporter=c_x, c_importer=c_m)
    assert chk["agreement"] is False


def test_greedy_fill_matches_the_lp_on_random_instances():
    """A21: the merit-order fill must reproduce the LP optimum exactly (objective value)."""
    import random
    from wedge.reclear import _fill, _solve
    rnd = random.Random(7)
    for _ in range(200):
        ux = [Unit(f"x{i}", rnd.uniform(0, 150), rnd.uniform(0, 1.2), 0, rnd.uniform(50, 500),
                   rnd.choice([0.0, rnd.uniform(0, 40)])) for i in range(rnd.randint(1, 5))]
        um = [Unit(f"m{i}", rnd.uniform(0, 150), rnd.uniform(0, 1.2), 0, rnd.uniform(50, 500),
                   rnd.choice([0.0, rnd.uniform(0, 40)])) for i in range(rnd.randint(1, 5))]
        dx = rnd.uniform(sum(u.min_mw for u in ux), sum(u.capacity_mw for u in ux) - 60)
        dm = rnd.uniform(sum(u.min_mw for u in um) + 60, sum(u.capacity_mw for u in um))
        f, eta = rnd.uniform(1, 50), rnd.uniform(0.95, 1.0)
        lp = _solve(ux, um, dx, dm, f, eta)
        gx, gm = _fill(ux, dx + f / eta), _fill(um, dm - f)
        if lp is None:
            assert gx is None or gm is None
            continue
        c = {u.name: u.srmc for u in ux + um}
        lp_cost = sum(c[n] * v for n, v in lp.items())
        gr_cost = sum(c[n] * v for n, v in {**gx, **gm}.items())
        assert gr_cost == pytest.approx(lp_cost, rel=1e-6, abs=1e-6)


def test_emission_change_sums_every_moving_unit_across_a_technology_boundary():
    """A29: exporter -0.4 MWh at EF 1 and -0.6 MWh at EF 0, importer +1 MWh at
    EF 0.2 -> actual change -0.2 tCO2 (aligned). The pre-A29 code took the largest responder and
    returned +0.2 (misdirected)."""
    x = [Unit("dirty", srmc=60.0, ef_el=1.0, output_mw=0.4, capacity_mw=0.4),
         Unit("clean", srmc=50.0, ef_el=0.0, output_mw=10.0, capacity_mw=10.0)]
    m = [Unit("gas", srmc=40.0, ef_el=0.2, output_mw=10.0, capacity_mw=20.0)]
    r = reclear_hour(x, m, flow_mw=5.0, eta=1.0, cap_flow=1e6)
    assert r.delta_emissions == pytest.approx(-0.2)
    assert r.sign == -1


def test_robust_delta_independent_corners_widen_a_gas_for_gas_interval():
    """A30: with JOINT corners a gas -> gas displacement has identical EFs on both sides and the
    interval collapses to the loss term; INDEPENDENT corners must widen it to straddle zero."""
    from wedge.reclear import robust_delta

    def gas(q):
        qq = None if q is None else q.get("gas")
        ef = 0.47 if qq is None else 0.61 - qq * (0.61 - 0.34)
        return ef

    bx = lambda q: [Unit("x_gas", srmc=50.0, ef_el=gas(q), output_mw=500.0, capacity_mw=1000.0)]  # noqa: E731
    bm = lambda q: [Unit("m_gas", srmc=50.0, ef_el=gas(q), output_mw=500.0, capacity_mw=1000.0)]  # noqa: E731
    joint = robust_delta(bx, bm, flow_mw=300.0, eta_range=(0.97, 0.97), independent=False)
    indep = robust_delta(bx, bm, flow_mw=300.0, eta_range=(0.97, 0.97), independent=True)
    assert joint is not None and indep is not None
    assert joint[0] < 0 and joint[1] < 0          # joint: charge lowers emissions (loss term only)
    assert indep[0] < 0 < indep[1]                # independent: sign not identified


def test_side_decomposition_adds_up_to_delta():
    """A33: delta_emissions = im_em - ex_em."""
    x = [Unit("dirty", srmc=60.0, ef_el=1.0, output_mw=0.4, capacity_mw=0.4),
         Unit("clean", srmc=50.0, ef_el=0.0, output_mw=10.0, capacity_mw=10.0)]
    m = [Unit("gas", srmc=40.0, ef_el=0.2, output_mw=10.0, capacity_mw=20.0)]
    r = reclear_hour(x, m, flow_mw=5.0, eta=1.0, cap_flow=1e6)
    assert r.ex_em == pytest.approx(0.4)
    assert r.im_em == pytest.approx(0.2)
    assert r.delta_emissions == pytest.approx(r.im_em - r.ex_em)
