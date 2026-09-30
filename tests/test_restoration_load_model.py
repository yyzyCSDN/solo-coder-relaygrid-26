from relaygrid.restoration.adaptive import AdaptiveRestorationPlanner
from relaygrid.restoration.load_model import RecoveryLoadModel
from relaygrid.restoration.sequence import CandidateStep


def test_pickup_changes_with_outage_duration_and_customer_type():
    model = RecoveryLoadModel()
    residential = model.pickup_kw("residential", 100, 4)
    commercial = model.pickup_kw("commercial", 100, 4)

    assert residential.kw > commercial.kw
    assert residential.basis[:2] == (
        "conservative_customer_prior:residential",
        "no_observed_recovery_curve",
    )
    assert "sqrt_outage_cold_load" in residential.basis


def test_observed_curve_updates_similar_customer_model():
    model = RecoveryLoadModel()
    model.observe_curve(
        "branch:a",
        "residential",
        4,
        100,
        ((0, 110), (1, 105), (2, 100)),
    )
    estimate = model.pickup_kw("residential", 100, 4)

    unobserved = RecoveryLoadModel().pickup_kw("residential", 100, 4)
    assert estimate.kw < unobserved.kw
    assert estimate.observed is False
    assert "observed_recovery_events=1" in estimate.basis
    assert "upper_90pct_confidence" in estimate.basis


def test_planner_recalculates_capacity_and_order_after_observed_step():
    planner = AdaptiveRestorationPlanner(145)
    candidates = [
        CandidateStep("b", 33, 10, 0.1, True, 33, "residential", 4),
        CandidateStep("a", 100, 10, 0.1, True, 100, "residential", 4),
    ]

    initial = planner.plan(candidates, 0)
    assert initial.next_step.candidate.branch_id == "a"
    assert {decision.candidate.branch_id for decision in initial.decisions} == {"a", "b"}

    first = initial.next_step.candidate
    after_observation = planner.complete(
        first,
        0,
        outage_h=4,
        observed_curve=((0, 101), (1, 100.37), (2, 100.14)),
        pending=[candidates[0]],
    )
    assert after_observation.committed_kw == 101
    assert after_observation.next_step.candidate.branch_id == "b"
    after_observation = planner.plan([candidates[0]], 2)

    assert after_observation.executed[0].branch_id == "a"
    assert abs(after_observation.committed_kw - 100.14) < 0.01
    assert abs(after_observation.remaining_capacity_kw - 44.86) < 0.01
    assert after_observation.next_step.candidate.branch_id == "b"


def test_incremental_telemetry_updates_capacity_and_future_order():
    planner = AdaptiveRestorationPlanner(145)
    first = CandidateStep("a", 100, 100, 0.1, True, 100, "residential", 4)
    second = CandidateStep("b", 33, 10, 0.1, True, 33, "residential", 4)

    planner.complete(first, 0, outage_h=4)
    for elapsed_h, kw in ((0, 101), (1, 100.37), (2, 100.14)):
        planner.add_telemetry("a", elapsed_h, kw)
    plan = planner.plan([second], 2)

    assert abs(plan.committed_kw - 100.14) < 0.01
    assert plan.next_step.candidate.branch_id == "b"


def test_missing_observation_uses_conservative_non_decaying_load():
    planner = AdaptiveRestorationPlanner(140)
    step = CandidateStep("a", 100, 10, 0.1, True, 100, "residential", 4)
    planner.complete(step, 0, outage_h=4)

    later = planner.plan([], 2)

    assert later.committed_kw == 130
    assert later.remaining_capacity_kw == 10
    assert later.executed_estimates[0].basis == (
        *planner.executed[0].estimate.basis,
        "no_observed_recovery_curve",
        "no_decay_credit_without_telemetry",
    )


def test_completed_steps_are_not_rolled_back_when_capacity_overloaded():
    planner = AdaptiveRestorationPlanner(120)
    step = CandidateStep("a", 100, 10, 0.1, True, 100, "residential", 4)
    planner.complete(step, 0, outage_h=4)

    later = planner.plan([], 1)

    assert [executed.branch_id for executed in later.executed] == ["a"]
    assert later.committed_kw == 130
    assert later.remaining_capacity_kw == -10
