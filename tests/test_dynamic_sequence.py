"""Tests for the adaptive restoration sequencer."""
from __future__ import annotations

import pytest

from relaygrid.loads.cold_load import (
    ColdLoadModel,
    CustomerType,
    LoadObservation,
    ClpuProfile,
)
from relaygrid.restoration.sequence import (
    DynamicSequence,
    LoadStep,
    dynamic_sequence,
)


def _fast_decay_model() -> ColdLoadModel:
    """A high-inrush fast-decay profile plus a flat no-inrush profile."""
    return ColdLoadModel({
        CustomerType.RESIDENTIAL: ClpuProfile(2.5, 1.0, 0.5, "test high-inrush fast-decay"),
        CustomerType.COMMERCIAL: ClpuProfile(1.0, 0.1, 1.0, "test flat no-inrush"),
    })


def test_steps_committed_in_order_without_rollback():
    steps = [
        LoadStep("A", 40, 10, CustomerType.RESIDENTIAL),
        LoadStep("B", 55, 5, CustomerType.COMMERCIAL),
    ]
    res = dynamic_sequence(
        steps, 100.0, outage_hours=1.0, step_interval_hours=0.5,
        max_hours=8.0, model=_fast_decay_model(),
    )
    ids = [s.block_id for s in res.steps]
    assert ids == ["A", "B"]  # A committed first, never rolled back
    assert res.unmet == ()


def test_remaining_capacity_recomputed_each_step():
    steps = [
        LoadStep("A", 40, 10, CustomerType.RESIDENTIAL),
        LoadStep("B", 55, 5, CustomerType.COMMERCIAL),
    ]
    res = dynamic_sequence(
        steps, 100.0, outage_hours=1.0, step_interval_hours=0.5,
        max_hours=8.0, model=_fast_decay_model(),
    )
    # each step records the capacity remaining after that commit
    assert res.steps[0].remaining_capacity_kw == pytest.approx(
        100.0 - res.steps[0].restore_kw, rel=1e-9
    )
    # B only fits after A's inrush has decayed, so its remaining is small
    assert res.steps[1].remaining_capacity_kw < res.steps[0].remaining_capacity_kw
    assert res.final_remaining_capacity_kw >= 0.0


def test_basis_recorded_per_step():
    steps = [LoadStep("A", 40, 10, CustomerType.RESIDENTIAL)]
    res = dynamic_sequence(steps, 100.0, outage_hours=1.0)
    assert res.steps[0].basis  # non-empty basis
    assert "factor" in res.steps[0].basis


def test_waiting_frees_capacity_after_inrush_decays():
    # A draws a large inrush that decays quickly; B is flat.  B does not fit
    # at t=0 but fits after A's inrush decays.
    steps = [
        LoadStep("A", 40, 10, CustomerType.RESIDENTIAL),
        LoadStep("B", 55, 5, CustomerType.COMMERCIAL),
    ]
    res = dynamic_sequence(
        steps, 100.0, outage_hours=1.0, step_interval_hours=0.5,
        max_hours=8.0, model=_fast_decay_model(),
    )
    assert [s.block_id for s in res.steps] == ["A", "B"]
    # B's restore is sized at its (flat) demand, not A's inrush
    assert res.steps[1].restore_kw == pytest.approx(55.0, rel=1e-9)


def test_unmet_when_capacity_insufficient_even_after_waiting():
    # B's demand grows with outage duration and never fits.
    steps = [
        LoadStep("A", 40, 10, CustomerType.RESIDENTIAL),
        LoadStep("B", 60, 5, CustomerType.COMMERCIAL),
    ]
    res = dynamic_sequence(
        steps, 100.0, outage_hours=1.0, step_interval_hours=0.5,
        max_hours=8.0, model=_fast_decay_model(),
    )
    assert "B" in res.unmet
    # A is still committed (no rollback) even though B could not be restored
    assert [s.block_id for s in res.steps] == ["A"]


def test_observations_lower_inrush_restore_more_load():
    # Diversified total (100 kW) is below capacity, but conservative inrush
    # (131 kW) is not.  Observations showing a lower inrush on A free enough
    # capacity to also restore B.
    steps = [
        LoadStep("A", 60, 10, CustomerType.RESIDENTIAL),
        LoadStep("B", 40, 5, CustomerType.RESIDENTIAL),
    ]
    # without observations: conservative inrush, B unmet
    conservative = dynamic_sequence(
        steps, 120.0, outage_hours=1.0, step_interval_hours=0.5, max_hours=8.0,
    )
    # with observations showing a much lower inrush on A
    obs = [
        LoadObservation("A", 0.0, 65.0),
        LoadObservation("A", 0.5, 62.0),
    ]
    observed = dynamic_sequence(
        steps, 120.0, observations=obs, outage_hours=1.0,
        step_interval_hours=0.5, max_hours=8.0,
    )
    assert observed.total_restored_kw > conservative.total_restored_kw
    assert "B" in [s.block_id for s in observed.steps]
    # the observed basis is recorded on A's step
    assert "observed" in observed.steps[0].basis


def test_total_restored_customers_sums_committed_steps():
    steps = [
        LoadStep("A", 40, 10, CustomerType.RESIDENTIAL, customers=500),
        LoadStep("B", 55, 5, CustomerType.COMMERCIAL, customers=300),
    ]
    res = dynamic_sequence(
        steps, 100.0, outage_hours=1.0, step_interval_hours=0.5,
        max_hours=8.0, model=_fast_decay_model(),
    )
    assert res.total_restored_customers == 800


def test_empty_steps_returns_empty_sequence():
    res = dynamic_sequence([], 100.0)
    assert res.steps == ()
    assert res.total_restored_kw == 0.0
    assert res.unmet == ()
    assert res.final_remaining_capacity_kw == pytest.approx(100.0)
