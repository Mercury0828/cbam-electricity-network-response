"""`reclear` — the small zonal re-clearing cross-check.

Deliberately small. It is a validation instrument, not a second research project, and it
must not grow into one. If it starts needing unit commitment, storage dynamics or a network
representation, the guide says to stop and record the limitation instead.

What it does, and only this:

1. Builds a two-zone economic dispatch from the OBSERVED generation of one border-hour.
2. Applies the **marginal** perturbation — one MWh less delivered across the border — which is
   the same quantity the headline estimand uses. 🔴 Guide is explicit that perturbing by
   the whole charge instead would be a *different estimand*: over a finite suppression the
   marginal technology and even the regime can change.
3. Reports two things, separately:
   - **agreement**: does the sign of the modelled emission response match the sign `mefband`
     assigned?
   - 🔴 **coverage**: is the unit that actually responded in the model INSIDE `mefband`'s
     candidate set? A responder outside the set is a coverage failure (kill criterion K5b),
     not a disagreement to be averaged away.

🔴 **What this does NOT validate, stated up front.** The dispatch stack is built from the same
fuel and carbon inputs as `mefband`. So it tests the **identification logic** — candidate-set
construction, the directional filter, the fallback clause, the interval rule — and it does
**not** test the inputs. A wrong fuel price fools both. Amendment A8 (a fabricated nuclear
band) would NOT have been caught here; amendment A9 (the fallback clause suppressing French
gas and hydro) WOULD have been, because the LP dispatches gas as the flexible marginal unit
and would have reported it as a responder outside the candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass

import pulp

from .mefband import CandidateSet, Tech, covers


@dataclass
class Unit:
    """One technology in one zone, as the dispatch model sees it."""
    name: str
    srmc: float            # EUR/MWh electrical, a POINT value for the LP
    ef_el: float           # tCO2/MWh electrical, a POINT value
    output_mw: float       # observed
    capacity_mw: float     # available (or installed as an upper bound)
    min_mw: float = 0.0
    intertemporal: bool = False   # storage / reservoir: moves energy between intervals


@dataclass
class ReclearResult:
    solved: bool
    delta_emissions: float          # tCO2 for a 1 MWh reduction in delivered import
    sign: int | None                # +1 charging raises emissions, -1 lowers, 0 neutral
    exporter_responder: str | None
    importer_responder: str | None
    exporter_delta_mw: float
    importer_delta_mw: float
    note: str = ""
    # A22: units tied at the responder's marginal cost that could equally have responded.
    exporter_tie: tuple = ()
    importer_tie: tuple = ()
    # A33: side decomposition for a 1 MWh reduction in delivered import (per MWh):
    #   ex_em = emissions AVOIDED at the exporter (= e_out/eta), im_em = emissions ADDED at the
    #   importer (= e_in); delta_emissions = im_em - ex_em.
    ex_em: float = 0.0
    im_em: float = 0.0


def _solve(exporter: list[Unit], importer: list[Unit], demand_x: float,
           demand_m: float, flow_mw: float, eta: float) -> dict[str, float] | None:
    """Minimise dispatch cost at a FIXED delivered import `flow_mw`.

    Energy balance, with demand held fixed on both sides (assumption A3):

        exporter:  sum(gen_x) = demand_x + flow/eta     (it must also generate the losses)
        importer:  sum(gen_m) = demand_m - flow         (imports displace local generation)
    """
    prob = pulp.LpProblem("reclear", pulp.LpMinimize)
    gx = {u.name: pulp.LpVariable(f"x_{u.name}", u.min_mw, u.capacity_mw) for u in exporter}
    gm = {u.name: pulp.LpVariable(f"m_{u.name}", u.min_mw, u.capacity_mw) for u in importer}

    prob += (pulp.lpSum(u.srmc * gx[u.name] for u in exporter)
             + pulp.lpSum(u.srmc * gm[u.name] for u in importer))
    prob += pulp.lpSum(gx.values()) == demand_x + flow_mw / eta
    prob += pulp.lpSum(gm.values()) == demand_m - flow_mw

    if pulp.LpStatus[prob.solve(pulp.PULP_CBC_CMD(msg=0))] != "Optimal":
        return None
    return {n: (v.value() or 0.0) for n, v in {**gx, **gm}.items()}


def _fill(units: list[Unit], demand: float) -> dict[str, float] | None:
    """Exact solution of  min sum(c_i g_i)  s.t.  sum(g_i) = demand,  lo_i <= g_i <= hi_i.

    A single equality constraint over box-bounded variables: the optimum is the merit-order
    fill - every unit at its lower bound, then raise the cheapest first. Ties are broken by
    name, deterministically. Returns None if demand is outside [sum lo, sum hi] (infeasible).
    """
    lo = sum(u.min_mw for u in units)
    hi = sum(u.capacity_mw for u in units)
    if demand < lo - 1e-6 or demand > hi + 1e-6:
        return None
    g = {u.name: u.min_mw for u in units}
    rest = demand - lo
    for u in sorted(units, key=lambda x: (x.srmc, x.name)):
        add = min(rest, u.capacity_mw - u.min_mw)
        g[u.name] += add
        rest -= add
        if rest <= 1e-12:
            break
    return g


def _solve_fast(exporter, importer, demand_x, demand_m, flow_mw, eta):
    """With the border flow FIXED the two zones decouple, so each is solved exactly by `_fill`.
    Equivalent to the LP in `_solve` (A21) and orders of magnitude faster."""
    gx = _fill(exporter, demand_x + flow_mw / eta)
    gm = _fill(importer, demand_m - flow_mw)
    if gx is None or gm is None:
        return None
    return {**gx, **gm}


def reclear_hour(exporter: list[Unit], importer: list[Unit], *, flow_mw: float,
                 eta: float, cap_flow: float, epsilon_mw: float = 1.0) -> ReclearResult:
    """Suppress `epsilon_mw` of delivered import and see what actually moves."""
    if flow_mw <= epsilon_mw:
        return ReclearResult(False, 0.0, None, None, None, 0.0, 0.0,
                             "flow too small for a marginal perturbation")

    # Local demand implied by the OBSERVED state, then held fixed (A3).
    demand_x = sum(u.output_mw for u in exporter) - flow_mw / eta
    demand_m = sum(u.output_mw for u in importer) + flow_mw

    d0 = _solve_fast(exporter, importer, demand_x, demand_m, flow_mw, eta)
    d1 = _solve_fast(exporter, importer, demand_x, demand_m, flow_mw - epsilon_mw, eta)
    if d0 is None or d1 is None:
        return ReclearResult(False, 0.0, None, None, None, 0.0, 0.0, "LP infeasible")

    ef = {u.name: u.ef_el for u in exporter + importer}

    def biggest(units: list[Unit]) -> tuple[str | None, float]:
        best, bd = None, 0.0
        for u in units:
            d = d1[u.name] - d0[u.name]
            if abs(d) > abs(bd) + 1e-9:
                best, bd = u.name, d
        return best, bd

    def tie_group(units: list[Unit], resp: str | None, down: bool) -> tuple:
        """Units at the responder's cost with room to move the same way: equally optimal (A4)."""
        if resp is None:
            return ()
        c0 = next(u.srmc for u in units if u.name == resp)
        g = []
        for u in units:
            if abs(u.srmc - c0) > 1e-9:
                continue
            room = (d0[u.name] > u.min_mw + 1e-9) if down else (d0[u.name] < u.capacity_mw - 1e-9)
            if room or u.name == resp:
                g.append(u.name)
        return tuple(sorted(g))

    xr, xd = biggest(exporter)
    mr, md = biggest(importer)
    xt, mt = tie_group(exporter, xr, down=True), tie_group(importer, mr, down=False)

    # 🔴 A23. Primary intertemporal policy (`unresolve`): if an equally optimal responder moves
    # energy between settlement intervals (storage, reservoir), the consequence lies outside the
    # time boundary (A5) and the hour gets NO sign. `reclear` previously ignored the policy.
    # 🔴 A29. The emission change is the sum over
    # EVERY unit that moves, not the largest responder's tie group: when the perturbation crosses a
    # technology boundary (e.g. exporter -0.4 MWh at EF 1 and -0.6 MWh at EF 0) the old code
    # returned the wrong sign. Ties (A4) are handled per moving unit: its moved volume may be
    # reallocated to any unit on the same side at the same SRMC with room in the same direction,
    # which gives an interval [lo, hi]; the hour is signed only if the interval does not straddle 0.
    it = {u.name for u in exporter + importer if u.intertemporal}
    moved = {n: d1[n] - d0[n] for n in d0 if abs(d1[n] - d0[n]) > 1e-9}
    lo = hi = 0.0
    tie_sets: list[tuple] = []
    for side in (exporter, importer):
        by_cost: dict[float, float] = {}
        for u in side:
            if u.name in moved:
                by_cost[u.srmc] = by_cost.get(u.srmc, 0.0) + moved[u.name]
        for c, vol in by_cost.items():
            down = vol < 0
            grp = [u.name for u in side if abs(u.srmc - c) < 1e-9 and (
                u.name in moved or
                ((d0[u.name] > u.min_mw + 1e-9) if down else (d0[u.name] < u.capacity_mw - 1e-9)))]
            tie_sets.append(tuple(sorted(grp)))
            if it & set(grp):
                return ReclearResult(True, 0.0, None, xr, mr, xd, md,
                                     "intertemporal responder (A5 time boundary)", xt, mt)
            vals = (vol * min(ef[n] for n in grp), vol * max(ef[n] for n in grp))
            lo += min(vals)
            hi += max(vals)
    lo, hi = lo / epsilon_mw, hi / epsilon_mw
    if (lo > 1e-9) != (hi > 1e-9) or (lo < -1e-9) != (hi < -1e-9):
        return ReclearResult(True, 0.0, None, xr, mr, xd, md,
                             "tie-ambiguous marginal responder (A4)", xt, mt)
    delta_e = sum(v * ef[n] for n, v in moved.items()) / epsilon_mw
    xnames = {u.name for u in exporter}
    ex_em = -sum(v * ef[n] for n, v in moved.items() if n in xnames) / epsilon_mw
    im_em = sum(v * ef[n] for n, v in moved.items() if n not in xnames) / epsilon_mw

    sign = 0 if abs(delta_e) < 1e-9 and abs(lo) < 1e-9 and abs(hi) < 1e-9 else \
        (1 if lo > 1e-9 else -1)
    return ReclearResult(True, delta_e, sign, xr, mr, xd, md, "", xt, mt, ex_em, im_em)


def check_against_mefband(res: ReclearResult, sigma: int | None,
                          c_exporter: CandidateSet,
                          c_importer: CandidateSet) -> dict:
    """The two answers the study design asks for, kept separate.

    🔴 `agreement` and `coverage_ok` are different questions. A coverage failure invalidates
    the determined label whatever the agreement rate says (K5b).
    """
    out = {"solved": res.solved, "reclear_sign": res.sign, "mefband_sigma": sigma}
    if not res.solved:
        out["agreement"] = None
        out["coverage_ok"] = None
        out["note"] = res.note
        return out

    out["agreement"] = (sigma is not None and res.sign is not None and res.sign == sigma)

    # A22: coverage holds if ANY tied, equally optimal responder is in the candidate set.
    def ok(cs, resp, tie):
        names = tie or ((resp,) if resp else ())
        return (not names) or any(covers(cs, n) for n in names)

    missed = []
    if not ok(c_exporter, res.exporter_responder, res.exporter_tie):
        missed.append(f"exporter:{res.exporter_responder}")
    if not ok(c_importer, res.importer_responder, res.importer_tie):
        missed.append(f"importer:{res.importer_responder}")

    out["coverage_ok"] = not missed
    out["responders_outside_candidate_set"] = missed
    return out


# Technologies that do not respond to a 1 MWh marginal signal. Their OBSERVED output is
# their dispatch; they run baseload or to a contract, not to the marginal price.
MUST_RUN_KEYS = ("nuclear", "biomass", "waste", "other")


def _fuel_class(tech: Tech) -> str:
    n = tech.name.lower()
    if "ccgt" in n or "gas" in n:
        return "gas"
    if "coal" in n or "lignite" in n or "oil" in n:
        return "solid"
    return "other"


def units_from(techs: dict[str, Tech], gen_hour: dict[str, float],
               cap: dict[str, float], p_co2: float,
               clearing_price: float | None = None,
               q: dict[str, float] | None = None) -> list[Unit]:
    """Build LP units from the same technology definitions `mefband` uses.

    Point values are the MIDPOINTS of the declared intervals.

    🔴 A12, 2026-09-21. The first version priced every cost-unidentified unit at its emission
    cost alone, i.e. zero for nuclear, biomass and hydro. The LP then re-optimised from
    scratch and loaded those "free" units first, so its base case bore no resemblance to the
    observed dispatch - e.g. it ramped Hungarian biomass up ~215 MW that the market was not
    dispatching, while the OBSERVED state had CCGT running at 163 MW. A cheap unit sitting
    below capacity while an expensive one runs is REVEALED to be constrained, not marginal.
    Two corrections, both about the validation instrument, neither touching `mefband`:

      * must-run units (nuclear, biomass, waste, other) are FIXED at their observed output -
        they run baseload or to contract and do not answer a 1 MWh marginal signal;
      * opportunity-cost bidders (hydro reservoir, pumped storage, battery - i.e. the
        `intertemporal` technologies ONLY) stay flexible and are priced at the CLEARING PRICE,
        which is what an opportunity-cost bid is;
      * thermal units with an unpublished fuel cost keep their CARBON COST as SRMC, a real
        lower bound. (A first draft priced them at the clearing price too; that was wrong
        and was narrowed after it produced spurious Matra-lignite coverage failures.)

    🔴 Shared-input limitation unchanged: this still inherits `mefband`'s fuel and carbon
    inputs, so it validates the identification LOGIC, not the inputs.
    """
    out = []
    for name, tech in techs.items():
        if name not in gen_hour:
            continue
        # q[class] in [0, 1] selects a point in the DECLARED efficiency range (A20). Cost and
        # emission factor move together: a less efficient unit is both costlier and dirtier,
        # EF_el = EF_th / eta. q=None reproduces the original midpoint behaviour exactly.
        qq = None if q is None else q.get(_fuel_class(tech))
        if qq is None:
            ef = sum(tech.ef_el_range) / 2.0
            eta_pt = sum(tech.eta_range) / 2.0
        else:
            eta_pt = tech.eta_range[0] + qq * (tech.eta_range[1] - tech.eta_range[0])
            ef = tech.ef_el_range[1] - qq * (tech.ef_el_range[1] - tech.ef_el_range[0])
        o = max(gen_hour[name], 0.0)
        tn = tech.name.lower()

        if tech.must_run or any(k in tn for k in MUST_RUN_KEYS):
            out.append(Unit(tech.name, 0.0, ef, o, o, o))       # fixed: lo = hi = observed
            continue

        if tech.cost_unidentified or tech.fuel_th is None:
            if tech.intertemporal and clearing_price is not None:
                # genuine opportunity-cost bidder (reservoir, pumped storage, battery):
                # it bids the value of stored energy, approximated by the market price.
                srmc = clearing_price
            else:
                # 🔴 A12 narrowed: a THERMAL unit with an unpublished fuel cost (e.g. Matra
                # lignite) is NOT an opportunity-cost bidder. Its carbon cost is a real,
                # computable lower bound on its SRMC, and for lignite under the EU ETS it
                # dominates (83.9 EUR/t x ~1.2 t/MWh ~ 100 EUR/MWh). Pricing it at the
                # clearing price instead - as the first A12 draft did - put it artificially
                # at the margin and generated spurious coverage failures.
                srmc = p_co2 * ef
        else:
            vom = sum(tech.vom_range) / 2.0
            srmc = tech.fuel_th / eta_pt + p_co2 * ef + vom
        upper = o if tech.variable else max(cap.get(name, o), o)
        # 🔴 A16. An OFFLINE thermal unit does not start up for a 1 MWh marginal signal; a
        # start-up is a commitment decision, not a marginal response, and this model has no
        # unit commitment. Without this, the LP ramped an
        # offline Matra lignite plant up from zero and reported it as the responder. `mefband`
        # already excludes offline units via its partial-loading test, so the two models now
        # agree on the physics. Threshold: below 2 % of capacity counts as offline.
        if (not tech.variable and not tech.intertemporal
                and tech.fuel_th is not None and o < 0.02 * max(upper, 1e-9)):
            out.append(Unit(tech.name, srmc, ef, o, o, o))      # held at current output
            continue
        out.append(Unit(tech.name, srmc, ef, o, upper, 0.0, tech.intertemporal))
    return out


# Declared corners of the pre-registered efficiency axis, per fuel class. A sign is
# DISPATCH-ROBUST only if every corner gives it. (gas, solid) quantiles; 0 = least efficient.
CORNERS = [None, {"gas": 0.0, "solid": 0.0}, {"gas": 1.0, "solid": 1.0},
           {"gas": 0.0, "solid": 1.0}, {"gas": 1.0, "solid": 0.0}]


# A30: tier-B sign robustness over JOINT corners (pre-registered, D-009) by default; the
# INDEPENDENT-corners variant (exporter and importer efficiencies varied separately) is the
# stricter sensitivity used as a sensitivity. Set by `run_tiers.py --independent`.
INDEPENDENT_SIGN_CORNERS = False


def robust_sign(build_x, build_m, *, flow_mw: float, eta_range: tuple[float, float]):
    """Sign of the dispatch response across all declared corners AND both loss-factor bounds.

    Returns (sign, n_scenarios) where sign is +1/-1 if every scenario agrees, else None.
    `build_x(q)` / `build_m(q)` return unit lists for corner q.
    """
    signs = set()
    n = 0
    pairs = ([(qx, qm) for qx in CORNERS for qm in CORNERS] if INDEPENDENT_SIGN_CORNERS
             else [(q, q) for q in CORNERS])
    for qx, qm in pairs:
        ux, um = build_x(qx), build_m(qm)
        for eta in eta_range:
            r = reclear_hour(ux, um, flow_mw=flow_mw, eta=eta, cap_flow=1e6)
            if not r.solved:
                return None, n
            n += 1
            signs.add(r.sign)
            if len(signs) > 1:
                return None, n
    s = signs.pop()
    return (s if s != 0 else None), n


def robust_delta(build_x, build_m, *, flow_mw: float, eta_range: tuple[float, float],
                 independent: bool = True):
    """Interval [lo, hi] of the emission change for a 1 MWh reduction in delivered import, over all
    declared corners AND both loss-factor bounds (the magnitude companion of `robust_sign`).

    🔴 A30: `independent=True` (default here) varies the exporter's and the
    importer's efficiency corners INDEPENDENTLY (5 x 5 x 2 = 50 scenarios). With joint corners a
    gas -> gas displacement gets identical EFs on both sides and kappa collapses mechanically to the
    loss term, which would overstate the precision of any magnitude.

    Returns None if any scenario is unsolved or unsigned (intertemporal / tie-ambiguous): a
    magnitude is reported only where every scenario produces one. Signs MAY differ across
    scenarios, so the interval can straddle 0 (this population is wider than tier B).
    """
    vals = []
    pairs = ([(qx, qm) for qx in CORNERS for qm in CORNERS] if independent
             else [(q, q) for q in CORNERS])
    for qx, qm in pairs:
        ux, um = build_x(qx), build_m(qm)
        for eta in eta_range:
            r = reclear_hour(ux, um, flow_mw=flow_mw, eta=eta, cap_flow=1e6)
            if not r.solved or r.sign is None:
                return None
            vals.append(r.delta_emissions)
    return min(vals), max(vals)


def robust_scenarios(build_x, build_m, *, flow_mw: float, eta_range: tuple[float, float],
                     independent: bool = True, keyed: bool = False):
    """All (ex_em, im_em) pairs over the declared scenarios (A33), or None if any scenario is
    unsolved or unsigned. Used for money-valued benchmarks that weight the two sides differently."""
    out = []
    pairs = ([(qx, qm) for qx in CORNERS for qm in CORNERS] if independent
             else [(q, q) for q in CORNERS])
    for qx, qm in pairs:
        ux, um = build_x(qx), build_m(qm)
        for eta in eta_range:
            r = reclear_hour(ux, um, flow_mw=flow_mw, eta=eta, cap_flow=1e6)
            if not r.solved or r.sign is None:
                return None
            out.append(((CORNERS.index(qx), CORNERS.index(qm), eta, r.ex_em, r.im_em)
                        if keyed else (r.ex_em, r.im_em)))
    return out
