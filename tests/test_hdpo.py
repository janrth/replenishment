from dataclasses import replace

import pytest

torch = pytest.importorskip('torch')

from replenishment import (
    InventoryState, RiskObjective, create_structured_neural_policy,
    differentiable_rollout, train_hdpo, tensor_risk_score, simulate_replenishment,
)
from replenishment.neural import _model_from_policy


@pytest.mark.parametrize('lead_time', [0, 1, 2])
@pytest.mark.parametrize('review', [1, 3])
def test_tensor_rollout_matches_exact_integer_python_timing(lead_time, review):
    policy = create_structured_neural_policy(lead_time=lead_time, target_stock=7,
        review_period=review, seed=4)
    demand = [2, 4, 1, 0, 3, 5]
    costs = dict(holding_cost_per_unit=.5, stockout_cost_per_unit=3, order_cost_per_unit=.1)
    tensor = differentiable_rollout(policy, [demand], initial_on_hand=1, **costs)
    exact = simulate_replenishment(periods=len(demand), demand=demand,
        initial_on_hand=1, lead_time=lead_time, policy=policy, **costs)
    assert tensor.costs.tolist() == pytest.approx([exact.summary.total_cost])
    assert tensor.orders[0].tolist() == [s.order_placed for s in exact.snapshots]
    assert tensor.receipts[0].tolist() == [s.received for s in exact.snapshots]
    assert tensor.on_hand[0].tolist() == [s.ending_on_hand for s in exact.snapshots]
    assert tensor.lost_sales[0].tolist() == [s.lost_sales for s in exact.snapshots]


def test_pathwise_gradient_matches_independent_finite_difference_away_from_kinks():
    policy = create_structured_neural_policy(lead_time=1, target_stock=2.3, residual_scale=2, seed=4)
    model = _model_from_policy(policy)
    def loss():
        return differentiable_rollout(policy, [[3, 3, 3, 3]], model=model,
            holding_cost_per_unit=.1, stockout_cost_per_unit=5).costs.mean()
    actual = loss()
    actual.backward()
    gradient = model[2].bias.grad.item()
    epsilon = 1e-5
    with torch.no_grad():
        model[2].bias.add_(epsilon)
        high = loss().item()
        model[2].bias.sub_(2*epsilon)
        low = loss().item()
    assert abs(gradient) > 1
    assert gradient == pytest.approx((high - low)/(2*epsilon), rel=1e-5)


@pytest.mark.parametrize('mode', ['mean', 'cvar', 'mean_cvar', 'worst_case'])
def test_tensor_risk_agrees_with_hand_objective_and_propagates_gradient(mode):
    costs = torch.tensor([0., 2., 4., 12.], dtype=torch.float64, requires_grad=True)
    objective = RiskObjective(mode, alpha=.625, risk_weight=.3)
    result = tensor_risk_score(costs, objective)
    assert result.item() == pytest.approx(objective.score(costs.detach().tolist()))
    result.backward()
    assert costs.grad.sum().item() == pytest.approx(1)
    if mode == 'cvar':
        assert costs.grad.tolist() == pytest.approx([0, 0, 1/3, 2/3])


def test_training_updates_and_selects_using_separate_exact_validation():
    policy = create_structured_neural_policy(lead_time=1, target_stock=1, residual_scale=5, seed=19)
    original = policy.weights
    before_rng = torch.get_rng_state().clone()
    args = dict(training_demand=[[3]*8, [2, 4]*4], validation_demand=[[3]*8],
        epochs=25, learning_rate=.04, stockout_cost_per_unit=10, holding_cost_per_unit=.1, seed=31)
    result = train_hdpo(policy, **args)
    repeated = train_hdpo(policy, **args)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert policy.weights == original
    assert result.policy.weights != original
    assert result.policy.weights == repeated.policy.weights
    assert result.validation_score <= result.history[0].validation_score
    assert result.validation_score == min(e.validation_score for e in result.history)
    assert result.selected_epoch > 0
    assert len(result.history) == 26
    assert result.training_exact_score - result.training_continuous_score == pytest.approx(result.rounding_gap)


def test_first_action_does_not_observe_future_realized_demand():
    policy = create_structured_neural_policy(lead_time=1, target_stock=3, seed=5)
    first = differentiable_rollout(policy, [[1, 2, 3]])
    changed_future = differentiable_rollout(policy, [[1, 50, 100]])
    assert first.orders[0, 0].item() == changed_future.orders[0, 0].item()


def test_hdpo_rejects_fixed_order_cost_and_missing_dynamic_forecasts():
    policy = create_structured_neural_policy(lead_time=1, target_stock=3, seed=2)
    with pytest.raises(ValueError, match='fixed|order_cost_per_order'):
        train_hdpo(policy, training_demand=[[2, 2]], validation_demand=[[2, 2]], order_cost_per_order=1)
    with pytest.raises(ValueError, match='forecast'):
        train_hdpo(policy.with_forecast(lambda s: [2]), training_demand=[[2, 2]], validation_demand=[[2, 2]])


@pytest.mark.parametrize('demand', [[], [[]], [[1, -1]], [[1, float('nan')]], [[1], [1, 2]]])
def test_tensor_rollout_validates_demand_paths(demand):
    policy = create_structured_neural_policy(lead_time=1, target_stock=3)
    with pytest.raises((TypeError, ValueError)):
        differentiable_rollout(policy, demand)


def test_validation_forecasts_match_exact_rounding_and_do_not_become_deployment_data():
    provider = lambda s: [2 + s.period % 2]
    policy = create_structured_neural_policy(lead_time=1, target_stock=1,
        forecast=provider, use_forecast_target=True, seed=2)
    forecasts = [[[2], [3], [2], [3]]]
    result = train_hdpo(policy, training_demand=[[2, 3, 2, 3]], validation_demand=[[2, 3, 2, 3]],
        training_forecasts=forecasts, validation_forecasts=forecasts, epochs=2,
        holding_cost_per_unit=.5, stockout_cost_per_unit=5)
    assert result.policy.forecast is provider
    exact = simulate_replenishment(periods=4, demand=[2, 3, 2, 3], initial_on_hand=0,
        lead_time=1, policy=result.policy, holding_cost_per_unit=.5, stockout_cost_per_unit=5)
    assert result.validation_score == exact.summary.total_cost
