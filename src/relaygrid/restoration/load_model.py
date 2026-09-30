from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from statistics import fmean, stdev


@dataclass(frozen=True)
class CustomerLoadProfile:
    customer_type: str
    diversity: float
    recovery_tau_h: float
    max_multiplier: float


@dataclass(frozen=True)
class LoadEstimate:
    kw: float
    multiplier: float
    basis: tuple[str, ...]
    observed: bool = False


_DEFAULT_PROFILES = (
    CustomerLoadProfile("residential", 0.15, 1.5, 3.0),
    CustomerLoadProfile("commercial", 0.10, 1.0, 2.5),
    CustomerLoadProfile("industrial", 0.07, 0.75, 2.0),
    CustomerLoadProfile("agricultural", 0.12, 2.0, 3.0),
    CustomerLoadProfile("critical", 0.12, 2.0, 3.0),
    CustomerLoadProfile("unknown", 0.25, 2.0, 4.0),
)

_TYPE_ALIASES = {
    "home": "residential",
    "household": "residential",
    "business": "commercial",
    "retail": "commercial",
    "manufacturing": "industrial",
    "farm": "agricultural",
    "hospital": "critical",
    "data_center": "critical",
    "water_treatment": "critical",
}


class RecoveryLoadModel:
    """Cold-load model with conservative priors and observed-curve updates."""

    def __init__(self, profiles: tuple[CustomerLoadProfile, ...] = _DEFAULT_PROFILES):
        self.profiles = {p.customer_type: p for p in profiles}
        if "unknown" not in self.profiles:
            raise ValueError("an unknown customer profile is required")
        self._events: dict[str, dict] = {}

    def observe_curve(
        self,
        event_id: str,
        customer_type: str,
        outage_h: float,
        base_kw: float,
        points: tuple[tuple[float, float], ...],
    ) -> None:
        previous_elapsed = None
        for elapsed_h, kw in sorted(points):
            if previous_elapsed is not None and elapsed_h <= previous_elapsed:
                raise ValueError("recovery observations must increase")
            self.observe_point(event_id, customer_type, outage_h, base_kw, elapsed_h, kw)
            previous_elapsed = elapsed_h

    def observe_point(
        self,
        event_id: str,
        customer_type: str,
        outage_h: float,
        base_kw: float,
        elapsed_h: float,
        kw: float,
    ) -> None:
        if outage_h < 0 or base_kw < 0 or elapsed_h < 0 or kw < 0:
            raise ValueError("recovery observation cannot be negative")
        profile = self._profile(customer_type)
        event = self._events.get(event_id)
        if event is None:
            event = {
                "customer_type": profile.customer_type,
                "outage_h": float(outage_h),
                "base_kw": float(base_kw),
                "points": [],
            }
            self._events[event_id] = event
        elif event["customer_type"] != profile.customer_type:
            raise ValueError("customer type cannot change for a recovery event")
        if event["points"] and elapsed_h <= event["points"][-1][0]:
            raise ValueError("recovery observations must increase")
        event["points"].append((float(elapsed_h), float(kw)))

    def pickup_multiplier(
        self, customer_type: str, outage_h: float, at_h: float = 0.0
    ) -> tuple[float, tuple[str, ...]]:
        if outage_h < 0 or at_h < 0:
            raise ValueError("time cannot be negative")
        profile = self._profile(customer_type)
        diversity, diversity_basis = self._conservative_diversity(profile)
        duration_h = max(0.0, outage_h + at_h)
        excess = min(profile.max_multiplier - 1.0, diversity * sqrt(duration_h))
        basis = diversity_basis + (
            f"outage_h={duration_h:.3g}",
            "sqrt_outage_cold_load",
        )
        return 1.0 + excess, basis

    def pickup_kw(
        self, customer_type: str, base_kw: float, outage_h: float, at_h: float = 0.0
    ) -> LoadEstimate:
        if base_kw < 0:
            raise ValueError("base load cannot be negative")
        multiplier, basis = self.pickup_multiplier(customer_type, outage_h, at_h)
        return LoadEstimate(max(0.0, base_kw * multiplier), multiplier, basis, False)

    def current_kw(
        self,
        customer_type: str,
        outage_h_at_pickup: float,
        base_kw: float,
        elapsed_h: float,
        observed_curve: tuple[tuple[float, float], ...] = (),
    ) -> LoadEstimate:
        if base_kw < 0 or elapsed_h < 0:
            raise ValueError("load and elapsed time cannot be negative")
        profile = self._profile(customer_type)
        pickup_multiplier, pickup_basis = self.pickup_multiplier(
            profile.customer_type, outage_h_at_pickup, 0.0
        )
        observed_points = tuple(
            (observed_at_h, kw)
            for observed_at_h, kw in sorted(observed_curve)
            if observed_at_h <= elapsed_h
        )
        points = observed_points
        if not points:
            basis = pickup_basis + (
                "no_observed_recovery_curve",
                "no_decay_credit_without_telemetry",
            )
            return LoadEstimate(base_kw * pickup_multiplier, pickup_multiplier, basis, False)

        ratios = (
            [kw / base_kw for _, kw in points]
            if base_kw > 0
            else [0.0 for _ in points]
        )
        ratio, basis = self._ratio_from_curve(
            profile, points, ratios, elapsed_h, pickup_multiplier
        )
        return LoadEstimate(max(0.0, base_kw * ratio), ratio, basis, True)

    def replace_observation_curve(
        self,
        event_id: str,
        customer_type: str,
        outage_h: float,
        base_kw: float,
        points: tuple[tuple[float, float], ...],
    ) -> None:
        self._events.pop(event_id, None)
        self.observe_curve(event_id, customer_type, outage_h, base_kw, points)

    def _profile(self, customer_type: str) -> CustomerLoadProfile:
        key = str(customer_type or "unknown").lower().replace(" ", "_")
        key = _TYPE_ALIASES.get(key, key)
        return self.profiles.get(key, self.profiles["unknown"])

    def _conservative_diversity(
        self, profile: CustomerLoadProfile
    ) -> tuple[float, tuple[str, ...]]:
        events = [
            event
            for event in self._events.values()
            if event["customer_type"] == profile.customer_type
            and event["outage_h"] > 0
            and event["points"]
        ]
        if not events:
            return profile.diversity, (
                f"conservative_customer_prior:{profile.customer_type}",
                "no_observed_recovery_curve",
            )

        implied = []
        for event in events:
            ratios = [
                kw / event["base_kw"] if event["base_kw"] > 0 else 0.0
                for _, kw in event["points"]
            ]
            amplitude, _, rmse = self._fit_decay(profile, event["points"], ratios)
            value = (amplitude + 1.28 * rmse) / sqrt(event["outage_h"])
            implied.append(min(profile.max_multiplier - 1.0, max(0.0, value)))

        observed_mean = fmean(implied)
        if len(implied) > 1:
            observed_upper = observed_mean + 1.28 * stdev(implied) / sqrt(len(implied))
        else:
            observed_upper = observed_mean
        # Two equivalent prior events keep a newly observed event from abruptly
        # reducing operating margin. The 90th-percentile estimate guards bias.
        weight = len(events) / (len(events) + 2)
        diversity = (1.0 - weight) * profile.diversity + weight * observed_upper
        diversity = min(profile.max_multiplier - 1.0, max(0.0, diversity))
        basis = (
            f"customer_prior:{profile.customer_type}",
            f"observed_recovery_events={len(events)}",
            "upper_90pct_confidence",
        )
        return diversity, basis

    def _fit_decay(
        self,
        profile: CustomerLoadProfile,
        points: list[tuple[float, float]],
        ratios: list[float],
    ) -> tuple[float, float, float]:
        # Values below connected-nominal load are treated as zero cold-load
        # excess rather than as evidence that cold load is negative.
        ys = [max(0.0, ratio - 1.0) for ratio in ratios]
        best = (float("inf"), 0.0, profile.recovery_tau_h)
        for tau in (
            profile.recovery_tau_h * factor
            for factor in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
        ):
            xs = [exp(-elapsed_h / tau) for elapsed_h, _ in points]
            denominator = sum(x * x for x in xs)
            amplitude = max(0.0, sum(y * x for x, y in zip(xs, ys)) / denominator)
            sse = sum((y - amplitude * x) ** 2 for x, y in zip(xs, ys))
            if sse < best[0]:
                best = (sse, amplitude, tau)
        sse, amplitude, tau = best
        rmse = sqrt(sse / len(ratios))
        return amplitude, tau, rmse

    def _ratio_from_curve(
        self,
        profile: CustomerLoadProfile,
        points: tuple[tuple[float, float], ...],
        ratios: list[float],
        elapsed_h: float,
        pickup_multiplier: float,
    ) -> tuple[float, tuple[str, ...]]:
        first_elapsed = points[0][0]
        if elapsed_h < first_elapsed:
            return max(pickup_multiplier, ratios[0]), (
                "conservative_pickup_before_first_observation",
            )
        if elapsed_h <= points[-1][0]:
            for (t1, _), (t2, _), r1, r2 in zip(points, points[1:], ratios, ratios[1:]):
                if t1 <= elapsed_h <= t2:
                    if t2 == t1:
                        ratio = r2
                    else:
                        ratio = r1 + (r2 - r1) * (
                            elapsed_h - t1
                        ) / (t2 - t1)
                    return max(0.0, ratio), ("observed_recovery_interpolation",)
            return ratios[-1], ("observed_recovery_latest",)

        last_ratio = ratios[-1]
        if len(points) == 1:
            return max(0.0, last_ratio), (
                "latest_observation_held",
                "no_decay_credit_without_a_curve",
            )

        _, tau, rmse = self._fit_decay(profile, points, ratios)
        last_elapsed = points[-1][0]
        remaining_cold_load = max(0.0, last_ratio - 1.0)
        projected = 1.0 + remaining_cold_load * exp(-(elapsed_h - last_elapsed) / tau)
        upper = projected + 1.28 * rmse
        ratio = max(projected, min(last_ratio, upper))
        if ratio == last_ratio:
            basis = (
                "latest_observation_caps_forecast",
                "observed_recovery_curve_fit",
                f"recovery_tau_h={tau:.3g}",
                "upper_90pct_confidence_extrapolation",
            )
        else:
            basis = (
                "observed_recovery_curve_fit",
                f"recovery_tau_h={tau:.3g}",
                "upper_90pct_confidence_extrapolation",
            )
        return max(0.0, ratio), basis
