from __future__ import annotations

from dataclasses import dataclass, replace

from relaygrid.restoration.load_model import LoadEstimate, RecoveryLoadModel
from relaygrid.restoration.sequence import CandidateStep


@dataclass(frozen=True)
class ExecutedRestorationStep:
    branch_id: str
    completed_at_h: float
    outage_h_at_pickup: float
    base_kw: float
    customer_type: str
    estimate: LoadEstimate
    observed_curve: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class AdaptiveStepDecision:
    candidate: CandidateStep
    estimate: LoadEstimate
    feasible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class AdaptiveRestorationPlan:
    capacity_kw: float
    committed_kw: float
    remaining_capacity_kw: float
    decisions: tuple[AdaptiveStepDecision, ...]
    executed: tuple[ExecutedRestorationStep, ...]
    executed_estimates: tuple[LoadEstimate, ...]

    @property
    def next_step(self) -> AdaptiveStepDecision | None:
        feasible = [decision for decision in self.decisions if decision.feasible]
        return feasible[0] if feasible else None


class AdaptiveRestorationPlanner:
    """Rank restoration steps, recalculating load and headroom after each step."""

    def __init__(self, capacity_kw: float, load_model: RecoveryLoadModel | None = None):
        if capacity_kw < 0:
            raise ValueError("capacity cannot be negative")
        self.capacity_kw = float(capacity_kw)
        self.load_model = load_model or RecoveryLoadModel()
        self._executed: dict[str, ExecutedRestorationStep] = {}
        self._last_completed_at_h = 0.0

    @property
    def executed(self) -> tuple[ExecutedRestorationStep, ...]:
        return tuple(
            sorted(self._executed.values(), key=lambda step: step.completed_at_h)
        )

    def plan(
        self,
        pending: list[CandidateStep],
        at_h: float = 0.0,
    ) -> AdaptiveRestorationPlan:
        if at_h < self._last_completed_at_h:
            raise ValueError("planning time cannot move backward")
        executed = self.executed
        executed_estimates = tuple(
            self._executed_load(step, at_h) for step in executed
        )
        committed_kw = sum(estimate.kw for estimate in executed_estimates)
        remaining_capacity_kw = self.capacity_kw - committed_kw
        seen = set(self._executed)
        duplicate = next(
            (candidate.branch_id for candidate in pending if candidate.branch_id in seen),
            None,
        )
        if duplicate is not None:
            raise ValueError(f"candidate {duplicate} has already executed")

        decisions = []
        for candidate in pending:
            base_kw = (
                candidate.base_kw
                if candidate.base_kw is not None
                else candidate.restored_kw
            )
            estimate = self.load_model.pickup_kw(
                candidate.customer_type,
                base_kw,
                self._outage_h(candidate, at_h),
            )
            reasons = self._feasibility_reasons(estimate, remaining_capacity_kw)
            decisions.append(
                AdaptiveStepDecision(candidate, estimate, not reasons, tuple(reasons))
            )

        infeasible = [decision for decision in decisions if not decision.feasible]
        ordered = self._rank_decisions(
            [decision for decision in decisions if decision.feasible]
        )
        ordered.extend(
            sorted(
                infeasible,
                key=lambda decision: (
                    decision.estimate.kw,
                    not decision.candidate.remote,
                    decision.candidate.branch_id,
                ),
            )
        )
        return AdaptiveRestorationPlan(
            self.capacity_kw,
            committed_kw,
            remaining_capacity_kw,
            tuple(ordered),
            executed,
            executed_estimates,
        )

    def next_step(
        self, pending: list[CandidateStep], at_h: float = 0.0
    ) -> AdaptiveStepDecision | None:
        return self.plan(pending, at_h).next_step

    def _rank_decisions(
        self, decisions: list[AdaptiveStepDecision]
    ) -> list[AdaptiveStepDecision]:
        def score(decision: AdaptiveStepDecision) -> float:
            candidate = decision.candidate
            benefit = candidate.restored_customers + 0.2 * decision.estimate.kw
            penalty = 1 + max(0.0, candidate.risk) * (
                1.5 if not candidate.remote else 1.0
            )
            return benefit / penalty

        return sorted(
            decisions,
            key=lambda decision: (
                -score(decision),
                not decision.candidate.remote,
                decision.candidate.branch_id,
            ),
        )

    def complete(
        self,
        step: CandidateStep,
        completed_at_h: float,
        outage_h: float | None = None,
        observed_curve: tuple[tuple[float, float], ...] = (),
        pending: list[CandidateStep] | None = None,
    ) -> AdaptiveRestorationPlan | None:
        if step.branch_id in self._executed:
            raise KeyError(step.branch_id)
        if completed_at_h < self._last_completed_at_h:
            raise ValueError("execution time cannot move backward")
        outage_duration = (
            self._outage_h(step, completed_at_h)
            if outage_h is None
            else outage_h
        )
        if outage_duration < 0:
            raise ValueError("outage duration cannot be negative")
        base_kw = (
            step.base_kw if step.base_kw is not None else step.restored_kw
        )
        estimate = self.load_model.pickup_kw(
            step.customer_type,
            base_kw,
            outage_duration,
        )
        event_id = f"branch:{step.branch_id}"
        if observed_curve:
            self.load_model.observe_curve(
                event_id,
                step.customer_type,
                outage_duration,
                base_kw,
                tuple(observed_curve),
            )
        self._executed[step.branch_id] = ExecutedRestorationStep(
            step.branch_id,
            float(completed_at_h),
            float(outage_duration),
            float(base_kw),
            step.customer_type,
            estimate,
            tuple(sorted(observed_curve)),
        )
        self._last_completed_at_h = float(completed_at_h)
        if pending is None:
            return None
        return self.plan(list(pending), completed_at_h)

    def add_telemetry(
        self,
        branch_id: str,
        elapsed_h: float,
        kw: float,
        pending: list[CandidateStep] | None = None,
    ) -> ExecutedRestorationStep | AdaptiveRestorationPlan:
        executed = self._executed.get(branch_id)
        if executed is None:
            raise KeyError(branch_id)
        if (
            executed.observed_curve
            and elapsed_h <= executed.observed_curve[-1][0]
        ):
            raise ValueError("telemetry observations must increase")
        point = (float(elapsed_h), float(kw))
        updated_curve = tuple(sorted(executed.observed_curve + (point,)))
        # Replace the event so the complete observation updates future estimates.
        event_id = f"branch:{branch_id}"
        self.load_model.replace_observation_curve(
            event_id,
            executed.customer_type,
            executed.outage_h_at_pickup,
            executed.base_kw,
            updated_curve,
        )
        self._executed[branch_id] = replace(executed, observed_curve=updated_curve)
        executed = self._executed[branch_id]
        if pending is not None:
            return self.plan(list(pending), executed.completed_at_h + elapsed_h)
        return executed

    def planned(
        self,
        step: CandidateStep,
        completed_at_h: float = 0.0,
        outage_h: float | None = None,
    ) -> ExecutedRestorationStep:
        decision = self.next_step([step], completed_at_h)
        if decision is None:
            raise ValueError("step is not feasible")
        self.complete(step, completed_at_h, outage_h)
        return self.executed[-1]

    def _executed_load(
        self, step: ExecutedRestorationStep, at_h: float
    ) -> LoadEstimate:
        return self.load_model.current_kw(
            step.customer_type,
            step.outage_h_at_pickup,
            step.base_kw,
            at_h - step.completed_at_h,
            step.observed_curve,
        )

    def _outage_h(self, candidate: CandidateStep, at_h: float) -> float:
        return max(0.0, candidate.outage_h + at_h)

    def _feasibility_reasons(
        self,
        estimate: LoadEstimate,
        remaining_capacity_kw: float,
    ) -> list[str]:
        reasons = []
        if estimate.kw > remaining_capacity_kw + 1e-9:
            reasons.append("insufficient_capacity")
        return reasons
