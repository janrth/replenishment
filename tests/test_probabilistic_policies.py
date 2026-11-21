import itertools
import math

import pytest

from replenishment import (
    InventoryState, ProjectedInventoryLevelPolicy, ProtectionPeriodQuantilePolicy,
    simulate_replenishment,
)


def test_pil_matches_independent_exhaustive_path_enumeration():
    state = InventoryState(5, 2, 5, 0, ((6, 1), (7, 4)))
    policy = ProjectedInventoryLevelPolicy(8, {0: 0.25, 3: 0.75}, lead_time=3)
    expected = 0.0
    for path in itertools.product([0, 3], repeat=2):
        stock = max(0, 2 + 1 - path[0])
        stock = max(0, stock + 4 - path[1])
        probability = math.prod(0.25 if d == 0 else 0.75 for d in path)
        expected += probability * stock
    assert expected == 2.5
    assert policy.projected_inventory(state) == pytest.approx(expected)
    assert policy.order_quantity_for(state) == 6


def test_pil_uses_receipt_timing_rather_than_total_pipeline():
    policy = ProjectedInventoryLevelPolicy(5, {3: 1.0}, lead_time=3)
    early = InventoryState(0, 0, 4, 0, ((1, 4),))
    late = InventoryState(0, 0, 4, 0, ((2, 4),))
    assert policy.projected_inventory(early) == 0
    assert policy.projected_inventory(late) == 1
    assert policy.order_quantity_for(early) == 5
    assert policy.order_quantity_for(late) == 4


@pytest.mark.parametrize('lead_time', [0, 1])
def test_pil_no_intervening_demand_for_zero_or_one_lead_time(lead_time):
    policy = ProjectedInventoryLevelPolicy(10, {20: 1.0}, lead_time)
    assert policy.projected_inventory(InventoryState(0, 3, 0, 0)) == 3
    assert policy.order_quantity_for(InventoryState(0, 3, 0, 0)) == 7


def test_pil_includes_old_receipt_at_arrival_before_arrival_demand():
    policy = ProjectedInventoryLevelPolicy(5, {3: 1.0}, lead_time=2)
    assert policy.projected_inventory(InventoryState(0, 1, 4, 0, ((2, 4),))) == 4


@pytest.mark.parametrize('schedule, on_order', [
    ((), 2), (((0, 2),), 2), (((4, 2),), 2), (((1, -2),), -2),
    (((2, 1), (1, 1)), 2), (((1, 1), (1, 1)), 2), (((1.5, 2),), 2),
])
def test_pil_rejects_inconsistent_or_invalid_receipt_schedule(schedule, on_order):
    policy = ProjectedInventoryLevelPolicy(5, {2: 1}, lead_time=3)
    with pytest.raises((ValueError, TypeError)):
        policy.projected_inventory(InventoryState(0, 0, on_order, 0, schedule))


@pytest.mark.parametrize('pmf', [{}, {-1: 1}, {1.5: 1}, {1: -1, 2: 2},
    {1: float('nan')}, {1: float('inf')}, {1: 0.5}, {True: 1}])
def test_pil_validates_discrete_distribution(pmf):
    with pytest.raises((ValueError, TypeError)):
        ProjectedInventoryLevelPolicy(5, pmf, 2)


def test_pil_freezes_pmf_and_review_cadence():
    pmf = {0: 0.5, 2: 0.5}
    policy = ProjectedInventoryLevelPolicy(5, pmf, 2, review_period=2)
    pmf.clear()
    assert policy.order_quantity_for(InventoryState(1, 0, 0, 0)) == 0
    assert policy.order_quantity_for(InventoryState(2, 0, 0, 0)) == 5


@pytest.mark.parametrize('policy', [
    ProjectedInventoryLevelPolicy(5, {2: 1}, lead_time=1),
    ProtectionPeriodQuantilePolicy(0.9, lead_time=1, aggregate_samples=[2]),
])
def test_simulator_rejects_policy_lead_time_mismatch_even_with_empty_pipeline(policy):
    with pytest.raises(ValueError, match='lead_time'):
        simulate_replenishment(periods=2, demand=[1, 2], initial_on_hand=0,
                               lead_time=3, policy=policy)


def test_aggregate_quantile_is_not_sum_of_marginal_quantiles():
    # Each marginal has a 50% quantile of zero, but path totals are all 10.
    policy = ProtectionPeriodQuantilePolicy(
        0.5, lead_time=2, review_period=1, demand_paths=[[0, 10], [10, 0]],
    )
    state = InventoryState(0, 2, 3, 0)
    assert policy.protection_period == 2
    assert policy.target_for(state) == 10
    assert policy.order_quantity_for(state) == 5


def test_joint_path_dependence_changes_quantile():
    state = InventoryState(0, 0, 0, 0)
    correlated = ProtectionPeriodQuantilePolicy(.75, 2, demand_paths=[[0, 0], [10, 10]])
    independent = ProtectionPeriodQuantilePolicy(
        .75, 2, demand_paths=list(itertools.product([0, 10], repeat=2)),
    )
    assert correlated.target_for(state) == 20
    assert independent.target_for(state) == 10


@pytest.mark.parametrize('q, expected', [(0, 0), (.5, 2), (.51, 8), (1, 10)])
def test_empirical_quantile_is_inverse_cdf_without_interpolation(q, expected):
    policy = ProtectionPeriodQuantilePolicy(q, 1, aggregate_samples=[10, 0, 8, 2])
    assert policy.target_for(InventoryState(0, 0, 0, 0)) == expected


@pytest.mark.parametrize('lead_time, review, horizon', [(0, 1, 1), (1, 1, 1), (2, 1, 2), (0, 3, 3), (2, 3, 4)])
def test_quantile_protection_period_follows_post_demand_timing(lead_time, review, horizon):
    policy = ProtectionPeriodQuantilePolicy(.5, lead_time, review, demand_paths=[[1] * horizon])
    assert policy.protection_period == horizon
    assert policy.order_quantity_for(InventoryState(0, 0, 0, 0)) == horizon
    if review > 1:
        assert policy.order_quantity_for(InventoryState(1, 0, 0, 0)) == 0


def test_quantile_freezes_static_samples_and_supports_as_of_provider():
    policy = ProtectionPeriodQuantilePolicy(.9, 1, aggregate_samples=iter([2, 4]))
    assert policy.target_for(InventoryState(0, 0, 0, 0)) == 4
    assert policy.target_for(InventoryState(1, 0, 0, 0)) == 4
    dynamic = ProtectionPeriodQuantilePolicy(.5, 1, aggregate_samples=lambda s: [s.period + 2])
    assert dynamic.target_for(InventoryState(3, 0, 0, 0)) == 5
    paths = [[1, 2]]
    frozen = ProtectionPeriodQuantilePolicy(.5, 2, demand_paths=paths)
    paths[0][0] = 100
    assert frozen.target_for(InventoryState(0, 0, 0, 0)) == 3


@pytest.mark.parametrize('kwargs', [
    {}, {'aggregate_samples': []}, {'aggregate_samples': [-1]},
    {'aggregate_samples': [float('nan')]}, {'aggregate_samples': [1.5]},
    {'aggregate_samples': [True]}, {'demand_paths': [[1]]},
    {'aggregate_samples': [2], 'demand_paths': [[1, 1]]},
])
def test_quantile_rejects_invalid_samples_and_shapes(kwargs):
    with pytest.raises((ValueError, TypeError)):
        ProtectionPeriodQuantilePolicy(.5, 2, **kwargs)


@pytest.mark.parametrize('q', [-.1, 1.1, float('nan'), float('inf'), True])
def test_quantile_rejects_invalid_probability(q):
    with pytest.raises((ValueError, TypeError)):
        ProtectionPeriodQuantilePolicy(q, 1, aggregate_samples=[1])


def test_provider_outputs_are_validated_at_decision_time():
    policy = ProtectionPeriodQuantilePolicy(.5, 2, demand_paths=lambda s: [[1]])
    with pytest.raises(ValueError, match='horizon'):
        policy.target_for(InventoryState(0, 0, 0, 0))
