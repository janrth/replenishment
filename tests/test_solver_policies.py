from dataclasses import FrozenInstanceError
import pytest

pytest.importorskip('scipy')

from replenishment import (
    DeterministicRollingHorizonPolicy, ScenarioRollingHorizonPolicy,
    SolverItemConfig, InventoryState, simulate_replenishment,
    simulate_replenishment_with_aggregation,
    generate_standard_simulation_rows, build_article_configs_from_standard_rows,
    simulate_replenishment_for_articles, build_replenishment_decisions_from_simulations,
)


def config(lead=1, review=1):
    return SolverItemConfig(lead, 8, holding_cost_per_unit=1,
                            stockout_cost_per_unit=10, review_period=review)


def test_deterministic_and_identical_scenarios_match_and_sources_are_copied():
    forecasts = [2, 3, 2]
    paths = [list(forecasts), list(forecasts)]
    a = DeterministicRollingHorizonPolicy(config(), forecasts)
    b = ScenarioRollingHorizonPolicy(config(), paths)
    forecasts[0] = 200
    paths[0][0] = 200
    assert a.order_quantity_for(InventoryState(0, 0, 0, 0)) == 2
    assert b.order_quantity_for(InventoryState(0, 0, 0, 0)) == 2
    assert a.forecast == (2, 3, 2)
    with pytest.raises(FrozenInstanceError):
        a.forecast = (1,)


def test_callback_is_asof_and_horizon_is_checked():
    seen = []
    def forecast(state):
        seen.append(state.period)
        return [state.period+1, 2]
    policy = DeterministicRollingHorizonPolicy(config(), forecast, horizon=2)
    result = simulate_replenishment(periods=3, demand=[1, 2, 3], initial_on_hand=0,
                                    lead_time=1, policy=policy)
    assert seen == [0, 1, 2]
    assert result.metadata.policy_name == 'DeterministicRollingHorizonPolicy'
    bad = DeterministicRollingHorizonPolicy(config(), lambda state: [1], horizon=2)
    with pytest.raises(ValueError, match='horizon'):
        bad.plan(InventoryState(0, 0, 0, 0))


@pytest.mark.parametrize('cls,source', [
    (DeterministicRollingHorizonPolicy, [2, 3, 2]),
    (ScenarioRollingHorizonPolicy, [[2, 3, 2], [2, 1, 3]]),
])
def test_standard_rows_decisions_and_review_cadence(cls, source):
    rows = generate_standard_simulation_rows(n_unique_ids=1, periods=6, seed=12,
        lead_time=0)
    policy = cls(config(0, 2), source)
    configs = build_article_configs_from_standard_rows(rows, policies={rows[0].unique_id: policy})
    results = simulate_replenishment_for_articles(configs)
    result = results[rows[0].unique_id]
    assert result.metadata.aggregation_window == 1
    assert result.metadata.review_period == 2
    assert [result.snapshots[k].order_placed for k in [1, 3, 5]] == [0, 0, 0]
    decisions = build_replenishment_decisions_from_simulations(rows, results)
    assert len(decisions) == 6
    for row, snapshot in zip(decisions, result.snapshots):
        assert row.quantity == snapshot.order_placed
    with pytest.raises(ValueError, match='aggregation|aggregation|unsupported',):
        simulate_replenishment_with_aggregation(periods=6, demand=[1]*6,
            initial_on_hand=0, lead_time=0, policy=policy, aggregation_window=2)


def test_policy_lead_time_mismatch_is_rejected_before_decision():
    with pytest.raises(ValueError, match='lead_time'):
        simulate_replenishment(periods=1, demand=[0], initial_on_hand=0, lead_time=2,
            policy=DeterministicRollingHorizonPolicy(config(), [2, 2]))


def test_callable_requires_explicit_horizon_and_static_horizon_must_match():
    with pytest.raises((ValueError, TypeError), match='horizon'):
        ScenarioRollingHorizonPolicy(config(), lambda state: [[2, 3]])
    with pytest.raises(ValueError, match='horizon'):
        DeterministicRollingHorizonPolicy(config(), [2, 3], horizon=3)


@pytest.mark.parametrize('horizon', [True, 2.0])
def test_static_horizon_must_be_an_integer(horizon):
    with pytest.raises(TypeError, match='horizon'):
        DeterministicRollingHorizonPolicy(config(), [1,1], horizon=horizon)


def test_daily_solver_forecasts_are_not_averaged_by_review_cadence():
    from dataclasses import replace
    rows = generate_standard_simulation_rows(n_unique_ids=1,periods=2,seed=4,lead_time=0)
    rows = [replace(r,forecast=f) for r,f in zip(rows,[2,10])]
    uid = rows[0].unique_id
    policy = DeterministicRollingHorizonPolicy(config(0,2),[2,10])
    results = simulate_replenishment_for_articles(
        build_article_configs_from_standard_rows(rows,policies={uid:policy}))
    decisions = build_replenishment_decisions_from_simulations(rows,results)
    assert [r.forecast_quantity for r in decisions] == [2,10]
    assert [r.forecast_quantity_lead_time for r in decisions] == [20,20]
    short = replace(results[uid],snapshots=results[uid].snapshots[:1])
    with pytest.raises(ValueError,match='snapshot|daily'):
        build_replenishment_decisions_from_simulations(rows,{uid:short})
