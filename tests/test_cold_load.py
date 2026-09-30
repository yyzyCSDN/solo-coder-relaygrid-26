"""Tests for the dynamic cold-load pickup model."""
from __future__ import annotations

import pytest

from relaygrid.loads.cold_load import (
    ColdLoadModel,
    CustomerType,
    LoadObservation,
    ClpuProfile,
    DEFAULT_PROFILES,
    outage_factor,
    restoration_factor,
)


def test_outage_factor_grows_with_outage_and_saturates():
    profile = DEFAULT_PROFILES[CustomerType.RESIDENTIAL]
    assert outage_factor(0.0, profile) == pytest.approx(1.0)
    # longer outage -> higher pickup, bounded by base_factor
    f_short = outage_factor(0.5, profile)
    f_long = outage_factor(10.0, profile)
    assert 1.0 < f_short < f_long < profile.base_factor
    assert f_long == pytest.approx(profile.base_factor, rel=0.02)


def test_restoration_factor_decays_toward_diversified_load():
    profile = DEFAULT_PROFILES[CustomerType.COMMERCIAL]
    m0 = outage_factor(2.0, profile)
    # at the instant of restoration the factor equals the outage pickup
    assert restoration_factor(2.0, 0.0, profile) == pytest.approx(m0)
    # long after restoration the inrush has decayed back to ~1.0
    assert restoration_factor(2.0, 10.0, profile) == pytest.approx(1.0, abs=0.02)
    # monotonic decay
    assert restoration_factor(2.0, 0.5, profile) > restoration_factor(2.0, 2.0, profile)


def test_conservative_no_observation_returns_type_profile_and_basis():
    model = ColdLoadModel()
    kw, factor, basis = model.demand_kw(
        "b1", 100.0, CustomerType.RESIDENTIAL, 1.0, 0.0, ()
    )
    expected = 100.0 * restoration_factor(1.0, 0.0, DEFAULT_PROFILES[CustomerType.RESIDENTIAL])
    assert kw == pytest.approx(expected, rel=1e-9)
    assert factor > 1.0
    # basis must document the conservative estimate and its origin
    assert "no observation" in basis
    assert "residential" in basis


def test_unknown_customer_type_uses_worst_case_conservative():
    model = ColdLoadModel()
    kw, factor, basis = model.demand_kw(
        "b2", 100.0, CustomerType.UNKNOWN, 2.0, 0.0, ()
    )
    # unknown type defaults to the highest base factor (2.0)
    assert factor == pytest.approx(restoration_factor(2.0, 0.0, DEFAULT_PROFILES[CustomerType.UNKNOWN]))
    assert factor >= 1.5
    assert "unknown" in basis


def test_observation_calibrates_factor_and_basis():
    model = ColdLoadModel()
    # observed 105 kW one hour after restoration
    obs = [LoadObservation("b1", 1.0, 105.0)]
    kw, factor, basis = model.demand_kw(
        "b1", 100.0, CustomerType.RESIDENTIAL, 1.0, 1.0, obs
    )
    conservative = restoration_factor(1.0, 1.0, DEFAULT_PROFILES[CustomerType.RESIDENTIAL])
    # the observation is lower than the conservative model -> calibrated factor drops
    assert factor < conservative
    assert kw == pytest.approx(100.0 * factor, rel=1e-9)
    assert "observed" in basis


def test_extrapolation_beyond_observation_floored_at_conservative_model():
    model = ColdLoadModel()
    # observation at t=0.5h; estimating at t=0h is backward extrapolation
    obs = [LoadObservation("b1", 0.5, 110.0)]
    kw, factor, basis = model.demand_kw(
        "b1", 100.0, CustomerType.RESIDENTIAL, 1.0, 0.0, obs
    )
    conservative = restoration_factor(1.0, 0.0, DEFAULT_PROFILES[CustomerType.RESIDENTIAL])
    # floored: never below the conservative model when extrapolating
    assert factor == pytest.approx(conservative, rel=1e-9)
    assert "extrapolated" in basis


def test_observations_for_other_blocks_are_ignored():
    model = ColdLoadModel()
    obs = [LoadObservation("other", 0.0, 50.0)]
    kw, factor, basis = model.demand_kw(
        "b1", 100.0, CustomerType.RESIDENTIAL, 1.0, 0.0, obs
    )
    assert "no observation" in basis
    assert kw == pytest.approx(
        100.0 * restoration_factor(1.0, 0.0, DEFAULT_PROFILES[CustomerType.RESIDENTIAL]),
        rel=1e-9,
    )


def test_custom_profile_overrides_default():
    custom = ClpuProfile(3.0, 2.0, 0.25, "custom test profile")
    model = ColdLoadModel({CustomerType.RESIDENTIAL: custom})
    kw, factor, basis = model.demand_kw(
        "b1", 100.0, CustomerType.RESIDENTIAL, 1.0, 0.0, ()
    )
    assert factor == pytest.approx(restoration_factor(1.0, 0.0, custom), rel=1e-9)
    assert "custom test profile" in basis
