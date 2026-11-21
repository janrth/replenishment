from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from replenishment import InventoryState, BaseStockPolicy, simulate_replenishment
from replenishment import (
    StructuredNeuralPolicy, create_structured_neural_policy,
    train_structured_neural_policy,
)


def test_core_and_star_imports_work_without_loading_torch():
    script = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError("blocked torch", name='torch')
sys.meta_path.insert(0, BlockTorch())
from replenishment import *
assert 'torch' not in sys.modules
assert BaseStockPolicy(3).order_quantity_for(InventoryState(0,0,0,0)) == 3
try:
    create_structured_neural_policy(lead_time=1)
except ImportError as e:
    assert 'advanced' in str(e)
else:
    raise AssertionError('Missing Torch was not reported')
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    subprocess.run([sys.executable, '-c', script], env=env, check=True, capture_output=True, text=True)


@pytest.fixture
def policy():
    pytest.importorskip('torch')
    return create_structured_neural_policy(lead_time=3, target_stock=10,
        residual_scale=5, feature_scale=10, hidden_size=4, seed=17)


def test_zero_residual_matches_base_stock_with_full_pipeline(policy):
    state = InventoryState(0, 2, 3, 0, ((1, 1), (2, 2)))
    assert policy.order_quantity_for(state) == BaseStockPolicy(10).order_quantity_for(state) == 5
    assert policy.continuous_order_for(state) == 5
    assert policy.features_for(state) == (.2, .1, .2, 0, 0, 0, 0)
    later = InventoryState(0, 2, 3, 0, ((2, 3),))
    assert policy.features_for(later) != policy.features_for(state)


def test_neural_review_cadence_and_forecast_target():
    pytest.importorskip('torch')
    policy = create_structured_neural_policy(lead_time=1, review_period=2,
        target_stock=1, use_forecast_target=True, forecast=[2, 4], seed=3)
    assert policy.order_quantity_for(InventoryState(0, 2, 0, 0)) == 5
    assert policy.order_quantity_for(InventoryState(1, 0, 0, 0)) == 0
    assert policy.with_forecast(lambda s: [s.period, 2]).order_quantity_for(InventoryState(2, 0, 0, 0)) == 5


def test_weights_are_immutable_and_checkpoint_roundtrip(policy, tmp_path):
    path = tmp_path / 'policy.json'
    policy.save(path)
    loaded = StructuredNeuralPolicy.load(path)
    state = InventoryState(0, 2, 3, 0, ((1, 3),))
    assert loaded == policy
    assert loaded.order_quantity_for(state) == 5
    document = json.loads(path.read_text())
    document['version'] = 999
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='version'):
        StructuredNeuralPolicy.load(path)


def test_external_forecast_is_required_when_loading_callable_checkpoint(policy, tmp_path):
    dynamic = policy.with_forecast(lambda s: [1, 2, 3])
    path = tmp_path / 'dynamic.json'
    dynamic.save(path)
    with pytest.raises(ValueError, match='forecast'):
        StructuredNeuralPolicy.load(path)
    loaded = StructuredNeuralPolicy.load(path, forecast=lambda s: [1, 2, 3])
    assert loaded.features_for(InventoryState(0, 0, 0, 0))[-3:] == (.1, .2, .3)


@pytest.mark.parametrize('kwargs', [
    {'lead_time': -1}, {'lead_time': True}, {'feature_scale': 0}, {'residual_scale': float('nan')},
    {'hidden_size': 0}, {'review_period': 0}, {'target_stock': -1}, {'forecast': [1]},
])
def test_invalid_neural_configuration_fails(kwargs):
    parameters = dict(lead_time=2)
    parameters.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        create_structured_neural_policy(**parameters)


def test_neural_adapter_rejects_inconsistent_schedule_and_bad_forecasts(policy):
    with pytest.raises(ValueError, match='on_order'):
        policy.order_quantity_for(InventoryState(0, 0, 4, 0))
    with pytest.raises(ValueError, match='forecast'):
        policy.with_forecast(lambda s: [1]).features_for(InventoryState(0, 0, 0, 0))
    with pytest.raises(ValueError):
        policy.with_forecast([1, -1, 2])


def test_supervised_training_learns_a_residual_without_mutating_input(policy):
    states = [InventoryState(0, stock, 0, 0) for stock in [0, 1, 2, 3]]
    original = policy.weights
    result = train_structured_neural_policy(policy, states=states,
        target_orders=[12, 11, 10, 9], epochs=80, learning_rate=.04, seed=5)
    assert policy.weights == original
    assert result.policy.weights != original
    assert result.loss_history[-1] < result.loss_history[0] * .1
    assert result.policy.order_quantity_for(states[0]) in {12, 13}


def test_neural_simulation_metadata_and_validation(policy):
    result = simulate_replenishment(periods=3, demand=[1, 2, 1], initial_on_hand=0,
        lead_time=3, policy=policy)
    assert result.metadata.policy_family == 'independent_item'
    assert result.metadata.policy_name == 'StructuredNeuralPolicy'
    with pytest.raises(ValueError, match='lead_time'):
        simulate_replenishment(periods=1, demand=[1], initial_on_hand=0, lead_time=1, policy=policy)
    with pytest.raises((ValueError, TypeError)):
        simulate_replenishment(periods=1, demand=[-1], initial_on_hand=0, lead_time=3, policy=policy)
