from __future__ import annotations
from dataclasses import dataclass

from relaygrid.loads.cold_load import (
    ColdLoadModel,
    CustomerType,
    LoadObservation,
)

@dataclass(frozen=True)
class CandidateStep:
    branch_id: str
    restored_kw: float
    restored_customers: int
    risk: float
    remote: bool

def rank_steps(steps: list[CandidateStep]):
    def score(s):
        benefit=s.restored_customers+0.2*s.restored_kw
        penalty=1+max(0.0,s.risk)*(1.5 if not s.remote else 1.0)
        return benefit/penalty
    return sorted(steps,key=lambda s:(-score(s),not s.remote,s.branch_id))

def sequence_until_target(steps: list[CandidateStep], target_customers: int):
    total=0; chosen=[]
    for s in rank_steps(steps):
        if total>=target_customers: break
        chosen.append(s); total+=s.restored_customers
    return chosen,total


@dataclass(frozen=True)
class LoadStep:
    """A load block that can be re-energized as a single restoration step."""
    block_id: str
    kw: float
    priority: int
    customer_type: CustomerType = CustomerType.UNKNOWN
    min_service_kw: float = 0.0
    risk: float = 0.0
    branch_id: str | None = None
    customers: int = 0


@dataclass(frozen=True)
class SequencedStep:
    step: int
    block_id: str
    branch_id: str | None
    restore_kw: float
    clpu_factor: float
    remaining_capacity_kw: float
    basis: str


@dataclass
class DynamicSequence:
    steps: tuple[SequencedStep, ...]
    total_restored_kw: float
    total_restored_customers: int
    unmet: tuple[str, ...]
    final_remaining_capacity_kw: float


def _step_score(step: LoadStep) -> float:
    benefit = step.customers + 0.2 * step.kw + 10.0 * step.priority
    penalty = 1.0 + max(0.0, step.risk)
    return benefit / penalty


def dynamic_sequence(
    steps: list[LoadStep],
    capacity_kw: float,
    *,
    observations: list[LoadObservation] | tuple[LoadObservation, ...] = (),
    outage_hours: float = 1.0,
    step_interval_hours: float = 0.25,
    max_hours: float = 24.0,
    model: ColdLoadModel | None = None,
) -> DynamicSequence:
    """Re-sequence restoration with a load model that updates every step.

    At each iteration the current demand of every block is recomputed from
    its outage duration, customer type, and the observed restoration curve;
    remaining capacity is recomputed; the next step is re-ranked; the best
    fitting step is committed and never rolled back.  When no step fits,
    time advances so the inrush on already-restored blocks can decay,
    freeing capacity.  Blocks still unrestored after ``max_hours`` are
    reported as unmet.

    Each committed step records the demand actually sized, the pickup
    factor, the remaining capacity, and the basis for the estimate
    (conservative type profile vs. observed calibration).
    """
    model = model or ColdLoadModel()
    obs = list(observations)
    by_id = {s.block_id: s for s in steps}
    restore_time: dict[str, float] = {}
    done: list[SequencedStep] = []
    now = 0.0

    def demand_at(block_id: str, base_kw: float, ctype, outage: float, since: float):
        return model.demand_kw(block_id, base_kw, ctype, outage, since, obs)

    def energized_at(t: float) -> float:
        total = 0.0
        for bid, t_rest in restore_time.items():
            s = by_id[bid]
            d, _, _ = demand_at(
                bid, s.kw, s.customer_type,
                outage_hours + t_rest, t - t_rest,
            )
            total += d
        return total

    while now <= max_hours:
        # Recompute remaining capacity from the current (decaying) inrush.
        remaining = capacity_kw - energized_at(now)

        # Re-rank candidates and pick the best one that fits.
        best = None
        for s in steps:
            if s.block_id in restore_time:
                continue
            outage_now = outage_hours + now
            d, f, basis = demand_at(s.block_id, s.kw, s.customer_type, outage_now, 0.0)
            if d > remaining + 1e-9:
                continue
            if s.min_service_kw > remaining + 1e-9:
                continue
            score = _step_score(s)
            if best is None or score > best[0] or (
                score == best[0] and s.block_id < best[1].block_id
            ):
                best = (score, s, d, f, basis)

        if best is None:
            # Nothing fits now; wait for inrush to decay on restored blocks.
            now += step_interval_hours
            continue

        _, s, d, f, basis = best
        restore_time[s.block_id] = now  # commit; executed steps are not rolled back
        done.append(SequencedStep(
            step=len(done) + 1,
            block_id=s.block_id,
            branch_id=s.branch_id,
            restore_kw=d,
            clpu_factor=f,
            remaining_capacity_kw=remaining - d,
            basis=basis,
        ))
        now += step_interval_hours

    restored_ids = set(restore_time)
    unmet = tuple(s.block_id for s in steps if s.block_id not in restored_ids)
    total_kw = sum(step.restore_kw for step in done)
    total_cust = sum(by_id[step.block_id].customers for step in done)
    return DynamicSequence(
        steps=tuple(done),
        total_restored_kw=total_kw,
        total_restored_customers=total_cust,
        unmet=unmet,
        final_remaining_capacity_kw=capacity_kw - energized_at(now),
    )
