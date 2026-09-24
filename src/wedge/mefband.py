"""`mefband` — interval identification of the marginal unit.

The marginal unit is not observable. This module therefore never
returns a point estimate. It returns a **candidate set** of technologies plus the emission
interval spanned by that set, so that a sign is decided only when every admissible pair on
both sides agrees.

Three things here are load-bearing and are unit-tested:

1. **The unit basis is converted exactly once.** With a thermal-basis fuel price and a
   thermal-basis emission factor,

       SRMC_k = fuel_th/eta_k + p_CO2 * EF_th_k/eta_k + VOM_k
              = fuel_th/eta_k + p_CO2 * EF_el_k      + VOM_k     since EF_el = EF_th/eta_k

   Dividing an electrical-basis `EF_el` by `eta_k` a second time is a unit error that shifts
   every band and corrupts both determinacy and the signs.

2. **The directional filter is one-sided per side, and the two sides need OPPOSITE
   directions.** To suppress one imported MWh the exporter must be able to *reduce* and the
   importer must be able to *increase*. Requiring both of a single technology would discard
   exactly the admissible responders.

3. **Emission factors are ranges, not point values.** Published technology factors are fleet
   averages; taking extrema over technology *identities* alone can manufacture apparent
   separation.

🔴 Coverage is not implied by the determinacy rate. A filter that is too narrow yields high
determinacy and confident wrong signs. Coverage is tested separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Direction(Enum):
    """Which way a technology must be able to move to be an admissible responder."""
    REDUCE = "reduce"      # exporter side, when an import is suppressed
    INCREASE = "increase"  # importer side, when an import is suppressed


class Label(str, Enum):
    """Evidence taxonomy, the study design"""
    DETERMINED_ALIGNED = "DETERMINED-ALIGNED"
    DETERMINED_MISDIRECTED = "DETERMINED-MISDIRECTED"
    NO_LOCAL_RESPONSE = "NO-LOCAL-RESPONSE"
    UNDETERMINED_SIGN = "UNDETERMINED-SIGN"
    UNRESOLVED_IDENTIFICATION = "UNRESOLVED-IDENTIFICATION"
    UNCLASSIFIED_REGIME = "UNCLASSIFIED-REGIME"
    OUT_OF_SCOPE = "OUT-OF-SCOPE"


@dataclass(frozen=True)
class Tech:
    """One generation technology in one zone.

    `ef_el_range` is the admissible range of the **electrical-basis** operating emission
    factor, tCO2 per MWh electrical. `ef_th` is the **thermal-basis** factor per MWh thermal
    input; exactly one of the two drives the carbon cost term and the conversion happens once.
    """
    name: str
    eta_range: tuple[float, float]          # conversion efficiency, el out per th in
    ef_el_range: tuple[float, float]        # tCO2 / MWh electrical
    vom_range: tuple[float, float] = (0.0, 0.0)   # EUR / MWh electrical
    fuel_th: float | None = None            # EUR / MWh thermal; None = not fuel-priced
    dispatchable: bool = True
    # opportunity-cost bidders (hydro, storage) cannot have an SRMC band built from fuel
    cost_unidentified: bool = False
    # intertemporal: moving energy between settlement intervals
    intertemporal: bool = False
    # variable renewable (wind, solar, run-of-river): output is set by the RESOURCE, not by
    # dispatch. It can be curtailed (REDUCE) but cannot be ramped UP on demand, however much
    # installed capacity sits idle. 🔴 Treating installed capacity as available upward
    # headroom for these is a physics error, corrected 2026-09-21 (amendment A10).
    variable: bool = False
    # must-run (nuclear, biomass, waste, other): runs baseload or to contract and does NOT
    # answer a 1 MWh marginal signal in either direction. Never a marginal responder. A13.
    must_run: bool = False

    @property
    def ef_th_range(self) -> tuple[float, float]:
        """Thermal-basis factor implied by the electrical-basis one. EF_th = EF_el * eta."""
        return (self.ef_el_range[0] * self.eta_range[0],
                self.ef_el_range[1] * self.eta_range[1])


@dataclass
class Observation:
    """What is observed for one technology in one zone in one hour."""
    tech: Tech
    output_mw: float
    available_capacity_mw: float
    technical_min_mw: float = 0.0
    interpolated: bool = False   # the study design rule 3: never used as evidence


def srmc_band(tech: Tech, p_co2: float) -> tuple[float, float] | None:
    """Short-run marginal cost band, EUR/MWh electrical.

    Returns None when no band can be constructed (opportunity-cost bidders, the study design's
    hydro/storage case). A None band widens the candidate set rather than narrowing it,
    which is the honest direction of the error.

    🔴 The carbon term uses `EF_el` directly. It is NOT divided by eta again.
    """
    if tech.cost_unidentified or tech.fuel_th is None:
        return None
    eta_lo, eta_hi = tech.eta_range
    if eta_lo <= 0 or eta_hi <= 0:
        raise ValueError(f"{tech.name}: efficiency must be positive")
    ef_lo, ef_hi = tech.ef_el_range
    vom_lo, vom_hi = tech.vom_range
    # fuel term falls as efficiency rises, so the low end pairs with eta_hi
    lo = tech.fuel_th / eta_hi + p_co2 * ef_lo + vom_lo
    hi = tech.fuel_th / eta_lo + p_co2 * ef_hi + vom_hi
    return (lo, hi)


def can_move(obs: Observation, direction: Direction) -> bool:
    """Is there adjustment room in the required direction?

    A unit at full capacity can still REDUCE. A unit at its technical minimum can still
    INCREASE. Requiring both would discard admissible responders, which is the error the
    guide explicitly flags.
    """
    if not obs.tech.dispatchable:
        return False
    if direction is Direction.REDUCE:
        return obs.output_mw > obs.technical_min_mw
    if obs.tech.variable:
        # resource-limited: already producing what the weather allows (A10)
        return False
    return obs.available_capacity_mw > obs.output_mw


@dataclass
class CandidateSet:
    zone: str
    direction: Direction
    members: list[Tech] = field(default_factory=list)
    reason: str = ""
    # technologies dropped only because their cost band excluded the clearing price
    price_excluded: list[str] = field(default_factory=list)
    # technologies that could move but whose band could not be built
    unidentified_cost: list[str] = field(default_factory=list)
    # marginal_v2 only: dropped because the unit was pinned at a bound (not price-setting)
    not_partially_loaded: list[str] = field(default_factory=list)
    # marginal_v2 only: cost-unidentified techs suppressed because the merit order explained
    # the price. 🔴 This list is the K5b exposure of the revised rule and must be audited.
    fallback_suppressed: list[str] = field(default_factory=list)
    has_intertemporal: bool = False

    @property
    def empty(self) -> bool:
        return not self.members

    def ef_bounds(self) -> tuple[float, float]:
        """(min lower bound, max upper bound) of EF_el over the set."""
        if self.empty:
            raise ValueError("empty candidate set has no emission interval")
        return (min(t.ef_el_range[0] for t in self.members),
                max(t.ef_el_range[1] for t in self.members))


def is_partially_loaded(obs: Observation, margin_frac: float = 0.02) -> bool:
    """Price-setting requires being able to move BOTH ways (rule `marginal_v2`).

    A unit pinned at its available capacity is infra-marginal: it would produce more if it
    could, so it is not setting the price. A unit at its technical minimum is extra-marginal.
    Only a partially loaded unit can be the price setter. This is the standard definition in
    the price-setting-technology literature (Blume-Werry et al. 2021; JRC JRC134300).

    🔴 This is a filter for MARGINAL-UNIT IDENTIFICATION, and it does NOT replace the
    one-directional `can_move` test, which answers a different question - whether a physical
    response is available at all. Guide step 2's warning against requiring both
    directions is about the response test, not about identifying the price setter.
    """
    cap = obs.available_capacity_mw
    if cap <= 0:
        return False
    m = margin_frac * cap
    return (obs.output_mw > obs.technical_min_mw + m) and (obs.output_mw < cap - m)


def build_candidates(zone: str, observations: list[Observation], *, clearing_price: float,
                     p_co2: float, direction: Direction,
                     price_tolerance: float,
                     candidate_rule: str = "guide_v1") -> CandidateSet:
    """Build `C(z, t)` for one zone-hour.

    `price_tolerance` is declared once, on defensible grounds, BEFORE results are looked at. It is never widened after seeing a determinacy rate.

    `candidate_rule` selects between two DECLARED variants, both runnable, so that any change
    reports before and after:

      "guide_v1"    - the study design as written: admit anything that can move in the required
                      direction; cost-unidentified technologies are always admitted.
                      🔴 Structurally yields 0% determinacy whenever both sides run a
                      zero-EF opportunity-cost bidder, which is most European hours (B16).

      "marginal_v2" - candidates must be PLAUSIBLY AT THE MARGIN:
                      (i)   partially loaded (see `is_partially_loaded`);
                      (ii)  cost-identified technologies must have the clearing price inside
                            their SRMC band, as before;
                      (iii) cost-UNIDENTIFIED technologies are a FALLBACK, admitted only when
                            no cost-identified technology explains the price. If the merit
                            order explains the price, it is used; if it does not, we admit
                            that we do not know and the unidentified ones enter.
                      🔴 Clause (iii) reintroduces a COVERAGE risk (K5b): it can exclude a
                      reservoir that truly was marginal in an hour where a CCGT band happens
                      to bracket the price. That risk is the reason the study design's re-clearing
                      cross-check is mandatory before any sign from this rule is reported.
    """
    if candidate_rule not in ("guide_v1", "marginal_v2", "marginal_v3"):
        raise ValueError("candidate_rule must be 'guide_v1', 'marginal_v2' or 'marginal_v3'")

    cs = CandidateSet(zone=zone, direction=direction)

    movable = []
    for obs in observations:
        if obs.interpolated:
            continue  # the study design rule 3 - interpolated values are not evidence
        if not can_move(obs, direction):
            continue
        if candidate_rule == "marginal_v3" and obs.tech.must_run:
            # must-run cannot respond to a marginal signal in either direction (A13)
            cs.not_partially_loaded.append(obs.tech.name)
            continue
        if candidate_rule in ("marginal_v2", "marginal_v3") and not is_partially_loaded(obs):
            cs.not_partially_loaded.append(obs.tech.name)
            continue
        movable.append(obs)

    if not movable:
        cs.reason = "no-adjustment-room"
        return cs

    identified: list[Tech] = []
    unidentified: list[Tech] = []
    for obs in movable:
        t = obs.tech
        band = srmc_band(t, p_co2)
        if band is None:
            unidentified.append(t)
            continue
        lo, hi = band
        if (lo - price_tolerance) <= clearing_price <= (hi + price_tolerance):
            identified.append(t)
        else:
            cs.price_excluded.append(t.name)

    if candidate_rule == "guide_v1":
        chosen = identified + unidentified
        cs.unidentified_cost.extend(t.name for t in unidentified)
    elif candidate_rule == "marginal_v2":
        if identified:
            chosen = identified
            cs.fallback_suppressed.extend(t.name for t in unidentified)
        else:
            chosen = unidentified
            cs.unidentified_cost.extend(t.name for t in unidentified)
    else:
        # marginal_v3 (A13). Split the cost-unidentified pool by PHYSICS:
        #   flexible opportunity-cost bidders (intertemporal: reservoir, pumped, battery)
        #     are ALWAYS admitted when partially loaded - they are frequently THE marginal
        #     unit (French reservoir hydro, found by reclear on GB->FR) and suppressing
        #     them is a coverage failure;
        #   other cost-unidentified units (thermal with unpublished fuel cost, e.g. lignite)
        #     keep the v2 fallback behaviour.
        # Must-run units never reach this point (filtered above).
        flexible = [t for t in unidentified if t.intertemporal]
        thermal = [t for t in unidentified if not t.intertemporal]
        cs.unidentified_cost.extend(t.name for t in flexible)
        if identified:
            chosen = identified + flexible
            cs.fallback_suppressed.extend(t.name for t in thermal)
        else:
            chosen = flexible + thermal
            cs.unidentified_cost.extend(t.name for t in thermal)

    cs.members.extend(chosen)
    cs.has_intertemporal = any(t.intertemporal for t in chosen)

    if cs.empty:
        # model and data disagree. Reported, never patched by widening the tolerance.
        cs.reason = "empty-candidate-set"
    return cs


def sign_from_sets(c_exporter: CandidateSet, c_importer: CandidateSet,
                   eta_link: float | tuple[float, float],
                   intertemporal_policy: str = "unresolve") -> tuple[int | None, str]:
    """The determinacy rule of the study design

        sigma = +1  if  min EF_el over C(m)  >  max EF_el over C(x) / eta
        sigma = -1  if  max EF_el over C(m)  <  min EF_el over C(x) / eta
        sigma = ?   otherwise

    Returns (sigma, reason). `sigma is None` means undetermined.

    🔴 `eta_link` is carried on the exporter side. Omitting it flips signs near the boundary. It may be a point value or an
    INTERVAL; where no border-specific loss factor is published, an interval is the honest
    representation and a sign is declared only if it holds for every admissible eta.
    """
    eta_lo, eta_hi = (eta_link, eta_link) if isinstance(eta_link, (int, float)) else eta_link
    if not (0.0 < eta_lo <= 1.0 and 0.0 < eta_hi <= 1.0 and eta_lo <= eta_hi):
        raise ValueError("eta_link must lie in (0, 1], and lo <= hi if an interval")
    if c_exporter.empty or c_importer.empty:
        return None, "empty-candidate-set"
    # Guide says an hour is unresolved where an intertemporal adjustment is "plausibly
    # LARGE" - a storage unit at the margin, a reservoir against a binding energy constraint.
    # Two admissible readings, both reported rather than one chosen silently:
    #   "unresolve"   - any intertemporal candidate present unresolves the hour. Stricter
    #                   than the guide's wording, and the conservative primary.
    #   "zero_factor" - the intertemporal candidate enters with its ~zero DIRECT operating
    #                   factor, which the guide also sanctions, and the time boundary (A5) is
    #                   stated in the text instead.
    if intertemporal_policy not in ("unresolve", "zero_factor"):
        raise ValueError("intertemporal_policy must be 'unresolve' or 'zero_factor'")
    if intertemporal_policy == "unresolve" and (
            c_exporter.has_intertemporal or c_importer.has_intertemporal):
        # energy moved between settlement intervals lies outside the time boundary (A5)
        return None, "intertemporal-adjustment-plausible"

    x_lo, x_hi = c_exporter.ef_bounds()
    m_lo, m_hi = c_importer.ef_bounds()

    # Interval arithmetic on eta. A sign is declared only if it holds for EVERY admissible
    # eta. For sigma = +1 the binding case is the SMALLEST eta (largest e_out/eta); for
    # sigma = -1 it is the LARGEST eta (smallest e_out/eta).
    if m_lo > x_hi / eta_lo:
        return +1, "importer strictly dirtier over the whole interval, for every admissible eta"
    if m_hi < x_lo / eta_hi:
        return -1, "importer strictly cleaner over the whole interval, for every admissible eta"
    return None, "emission intervals overlap"


def covers(candidate_set: CandidateSet, true_responder: str) -> bool:
    """Coverage test.

    The extrema in the determinacy rule are conservative ONLY if the technology that actually
    responded is inside the candidate set. This is a separate question from the determinacy
    rate and a high rate is not evidence for it.
    """
    return any(t.name == true_responder for t in candidate_set.members)
