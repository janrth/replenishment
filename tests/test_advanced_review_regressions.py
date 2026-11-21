import math

import pytest

torch = pytest.importorskip('torch')

from replenishment import (
    InventoryState, RiskObjective, empirical_cvar, create_structured_neural_policy,
    train_structured_neural_policy, train_hdpo, differentiable_rollout,
    tensor_risk_score, simulate_replenishment,
)


def test_supervised_learning_can_start_from_default_zero_target():
    policy = create_structured_neural_policy(lead_time=1, seed=3)
    result = train_structured_neural_policy(policy, states=[InventoryState(0, 0, 0, 0)],
        target_orders=[3], epochs=30, learning_rate=.025)
    assert result.policy.weights != policy.weights
    assert result.loss_history[-1] < 1


def test_hdpo_can_start_from_default_zero_target():
    policy = create_structured_neural_policy(lead_time=1, seed=3)
    result = train_hdpo(policy, training_demand=[[3]*5], validation_demand=[[3]*5],
        stockout_cost_per_unit=8, epochs=20, learning_rate=.025)
    assert result.selected_epoch > 0
    assert result.validation_score < 120


def test_supervised_learning_can_move_inward_from_the_order_cap():
    policy = create_structured_neural_policy(lead_time=1, target_stock=5, max_order_quantity=5, seed=3)
    result = train_structured_neural_policy(policy, states=[InventoryState(0,0,0,0)],
        target_orders=[3], epochs=30, learning_rate=.025)
    assert result.loss_history[-1] < 1


@pytest.mark.parametrize('forecast', [None, [0]])
def test_supervised_rejects_forecasts_incompatible_with_static_deployment(forecast):
    policy = create_structured_neural_policy(lead_time=1, forecast=forecast)
    with pytest.raises(ValueError, match='forecast'):
        train_structured_neural_policy(policy, states=[InventoryState(0,0,0,0)],
            target_orders=[3], forecasts=[[5]], epochs=1)


def test_hdpo_rejects_contradictory_static_forecast_override():
    policy = create_structured_neural_policy(lead_time=1, forecast=[0],
        use_forecast_target=True, residual_scale=0)
    with pytest.raises(ValueError, match='forecast'):
        train_hdpo(policy, training_demand=[[0,5]], validation_demand=[[0,5]],
            training_forecasts=[[[5],[0]]], validation_forecasts=[[[5],[0]]],
            stockout_cost_per_unit=1, epochs=1)


@pytest.mark.parametrize('objective', [
    RiskObjective('cvar'), RiskObjective('worst_case'), RiskObjective('mean_cvar', risk_weight=.5),
])
def test_active_tail_objectives_reject_partial_scenario_batches(objective):
    policy = create_structured_neural_policy(lead_time=1, target_stock=2)
    with pytest.raises(ValueError, match='batch|scenario'):
        train_hdpo(policy, training_demand=[[2,2],[2,5]], validation_demand=[[2,3]],
            objective=objective, batch_size=1, epochs=1)


def test_forecast_features_beyond_protection_period_do_not_inflate_baseline_target():
    policy = create_structured_neural_policy(lead_time=1, forecast_horizon=3,
        forecast=[0,0,100], use_forecast_target=True, seed=3)
    state = InventoryState(0,0,0,0)
    assert policy.features_for(state)[-1] == 10
    assert policy.order_quantity_for(state) == 0
    rollout = differentiable_rollout(policy, [[0,0]])
    assert rollout.orders.tolist() == [[0,0]]
    # Supervised targets also use the same one-period prefix baseline.
    result = train_structured_neural_policy(policy, states=[state], target_orders=[0], epochs=1)
    assert result.loss_history[0] == 0


def test_forecast_target_rejects_feature_horizon_shorter_than_protection():
    with pytest.raises(ValueError, match='horizon|protection'):
        create_structured_neural_policy(lead_time=2, forecast_horizon=1,
            use_forecast_target=True, forecast=[2])


def test_initialization_does_not_call_the_all_device_global_seeder(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('All-device seeding changes caller accelerator streams')
    monkeypatch.setattr(torch, 'manual_seed', forbidden)
    policy = create_structured_neural_policy(lead_time=1, target_stock=3, seed=17)
    assert policy.order_quantity_for(InventoryState(0,0,0,0)) == 3


def test_cpu_training_ignores_callers_default_tensor_device():
    # meta exposes accidental reliance on the caller's tensor device without
    # needing a GPU. All internal training allocations must explicitly use CPU.
    with torch.device('meta'):
        policy = create_structured_neural_policy(lead_time=1, target_stock=1, seed=17)
        result = train_hdpo(policy, training_demand=[[2,2]], validation_demand=[[2,2]],
            stockout_cost_per_unit=3, epochs=1)
        supervised = train_structured_neural_policy(policy,
            states=[InventoryState(0,0,0,0)], target_orders=[2], epochs=1)
    assert math.isfinite(result.validation_score)
    assert math.isfinite(supervised.loss_history[0])


@pytest.mark.parametrize('values, expected', [
    ([1e308, -1e308], 0), ([1e308,1e308], 1e308), ([1e16,1,-1e16], 1/3),
])
def test_finite_risk_means_and_zero_confidence_cvar_do_not_overflow_or_cancel(values, expected):
    for objective in [RiskObjective('mean'), RiskObjective('cvar', alpha=0)]:
        assert objective.score(values) == pytest.approx(expected)
        tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        score = tensor_risk_score(tensor, objective)
        assert score.item() == pytest.approx(expected)
        score.backward()
        assert tensor.grad.tolist() == pytest.approx([1/len(values)]*len(values))
    assert empirical_cvar(values, 0) == pytest.approx(expected)
