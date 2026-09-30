"""Dynamic cold-load pickup model.

Restoration sequencing historically used a fixed ``cold_load_multiplier`` on
every load block.  The inrush a block actually draws when re-energized
depends on three things:

* how long it was outaged (thermostatic load diversity is lost over time),
* the customer type (residential / commercial / industrial / critical), and
* the observed restoration curve (telemetry from previous energizations).

When no observation is available the model falls back to a conservative
type-based profile and reports the basis for the estimate alongside the
value.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import statistics


class CustomerType(str, Enum):
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClpuProfile:
    """Cold-load pickup parameters for a customer class.

    Attributes:
        base_factor: long-outage pickup multiplier (kW at restoration /
            diversified kW).
        outage_growth_per_hour: rate at which pickup grows with outage
            duration as load diversity is lost.
        decay_hours: post-restoration inrush decay time constant.
        basis: documented basis for the default parameters.
    """
    base_factor: float
    outage_growth_per_hour: float
    decay_hours: float
    basis: str


# Defaults follow utility CLPU rule-of-thumb ranges (EPRI/IEEE distribution
# restoration practice): residential ~1.5-2.0, commercial ~2.0-2.5,
# industrial ~1.3-1.5 (motor inrush dominates), critical facilities ~2.0.
# With no telemetry we use the conservative (upper) end of each range.
DEFAULT_PROFILES: dict[CustomerType, ClpuProfile] = {
    CustomerType.RESIDENTIAL: ClpuProfile(
        1.8, 0.50, 1.0,
        "no observation: residential CLPU 1.5-2.0 per utility rule-of-thumb; conservative upper bound",
    ),
    CustomerType.COMMERCIAL: ClpuProfile(
        2.2, 0.75, 0.75,
        "no observation: commercial CLPU 2.0-2.5 per utility rule-of-thumb; conservative upper bound",
    ),
    CustomerType.INDUSTRIAL: ClpuProfile(
        1.4, 0.35, 1.5,
        "no observation: industrial CLPU 1.3-1.5 (motor inrush); conservative upper bound",
    ),
    CustomerType.CRITICAL: ClpuProfile(
        2.0, 0.50, 1.0,
        "no observation: critical-facility CLPU ~2.0; conservative estimate",
    ),
    CustomerType.UNKNOWN: ClpuProfile(
        2.0, 0.50, 1.0,
        "no observation and unknown customer type: worst-case conservative default (factor 2.0)",
    ),
}


@dataclass(frozen=True)
class LoadObservation:
    """A point on the observed restoration curve of a load block.

    Attributes:
        block_id: the load block this observation belongs to.
        hours_since_restoration: hours after the block was re-energized
            (0 = instant of restoration).
        observed_kw: metered demand at that time.
    """
    block_id: str
    hours_since_restoration: float
    observed_kw: float


def outage_factor(outage_hours: float, profile: ClpuProfile) -> float:
    """Pickup multiplier at the instant of restoration.

    Grows from 1.0 (no outage) to ``profile.base_factor`` (long outage) as
    load diversity is lost.
    """
    h = max(0.0, outage_hours)
    return 1.0 + (profile.base_factor - 1.0) * (
        1.0 - math.exp(-profile.outage_growth_per_hour * h)
    )


def restoration_factor(
    outage_hours: float,
    hours_since_restoration: float,
    profile: ClpuProfile,
) -> float:
    """Pickup multiplier ``hours_since_restoration`` after re-energization.

    The inrush decays exponentially back toward 1.0 (normal diversified
    load).
    """
    m0 = outage_factor(outage_hours, profile)
    tau = max(0.0, hours_since_restoration)
    return 1.0 + (m0 - 1.0) * math.exp(-tau / profile.decay_hours)


class ColdLoadModel:
    """Computes the current demand of a load block.

    The estimate combines the type-based profile with any observed
    restoration curve.  When observations are missing the conservative
    type-based value is used and the basis is returned alongside.
    """

    def __init__(self, profiles: dict[CustomerType, ClpuProfile] | None = None):
        self.profiles = dict(DEFAULT_PROFILES)
        if profiles:
            self.profiles.update(profiles)

    def profile(self, customer_type: CustomerType) -> ClpuProfile:
        return self.profiles.get(customer_type, self.profiles[CustomerType.UNKNOWN])

    def demand_kw(
        self,
        block_id: str,
        base_kw: float,
        customer_type: CustomerType,
        outage_hours: float,
        hours_since_restoration: float = 0.0,
        observations: list[LoadObservation] | tuple[LoadObservation, ...] = (),
    ) -> tuple[float, float, str]:
        """Return ``(demand_kw, factor, basis)`` for a block at a point in time.

        ``basis`` documents whether the value came from the conservative
        type profile or from a calibration against observed restoration
        points, and is recorded per restoration step.
        """
        profile = self.profile(customer_type)
        model_factor = restoration_factor(outage_hours, hours_since_restoration, profile)

        if base_kw <= 0:
            return 0.0, 1.0, "no load: base_kw <= 0"

        block_obs = sorted(
            (o for o in observations if o.block_id == block_id and o.observed_kw > 0),
            key=lambda o: o.hours_since_restoration,
        )
        if not block_obs:
            basis = (
                f"{profile.basis}; outage={outage_hours:.2f}h, "
                f"restored={hours_since_restoration:.2f}h -> factor {model_factor:.2f}"
            )
            return base_kw * model_factor, model_factor, basis

        factor, note = self._calibrated_factor(
            profile, block_obs, base_kw, outage_hours, hours_since_restoration, model_factor
        )
        basis = (
            f"{note}; outage={outage_hours:.2f}h, restored={hours_since_restoration:.2f}h "
            f"-> factor {factor:.2f}"
        )
        return base_kw * factor, factor, basis

    def _calibrated_factor(
        self,
        profile: ClpuProfile,
        block_obs: list[LoadObservation],
        base_kw: float,
        outage_hours: float,
        hours_since_restoration: float,
        model_factor: float,
    ) -> tuple[float, str]:
        usable: list[tuple[float, float]] = []
        for o in block_obs:
            if o.observed_kw <= 0:
                continue
            o_factor = o.observed_kw / base_kw
            if o_factor > 1.0001:  # observation still carries pickup information
                usable.append((o.hours_since_restoration, o_factor))

        if not usable:
            return model_factor, (
                "observations show no pickup; conservative model retained"
            )

        # Fit the initial inrush amplitude m0 to the observations while keeping
        # the customer-type decay shape:
        #   o_factor = 1 + (m0 - 1) * exp(-tau / decay_hours)
        #   => m0 = 1 + (o_factor - 1) * exp(tau / decay_hours)
        fitted_m0s = [
            1.0 + (o_factor - 1.0) * math.exp(tau / profile.decay_hours)
            for tau, o_factor in usable
        ]
        fitted_m0 = max(1.0, statistics.median(fitted_m0s))
        factor = 1.0 + (fitted_m0 - 1.0) * math.exp(
            -hours_since_restoration / profile.decay_hours
        )

        tau_min = min(t for t, _ in usable)
        tau_max = max(t for t, _ in usable)
        extrapolating = hours_since_restoration > tau_max or hours_since_restoration < tau_min
        if extrapolating:
            # Outside the observed range we have no data to size the inrush;
            # floor the estimate at the conservative model so a single reading
            # cannot drive an unsafe (low) extrapolation.
            factor = max(factor, model_factor)
            note = (
                f"observed: {len(usable)} point(s), fitted initial pickup {fitted_m0:.2f}; "
                f"extrapolated outside [{tau_min:.2f},{tau_max:.2f}]h "
                f"-> floored at conservative model"
            )
        else:
            note = (
                f"observed: {len(usable)} point(s), fitted initial pickup {fitted_m0:.2f}"
            )
        return factor, note
