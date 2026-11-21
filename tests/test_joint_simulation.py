from dataclasses import replace
import pytest

pytest.importorskip('scipy')

from replenishment import (
    ArticleSimulationConfig, DeterministicRollingHorizonPolicy,
    ScenarioRollingHorizonPolicy, SolverItemConfig, SharedResourceLimits,
    simulate_replenishment_for_articles, simulate_coordinated_replenishment,
    generate_standard_simulation_rows, build_article_configs_from_standard_rows,
    build_replenishment_decisions_from_simulations, RiskObjective,
)


def article(lead=1, review=1, penalty=10, stock=0):
    item = SolverItemConfig(lead, 5, holding_cost_per_unit=1,
        stockout_cost_per_unit=penalty, order_cost_per_order=1,
        order_cost_per_unit=.5, review_period=review)
    return ArticleSimulationConfig(6, [1, 3, 2, 0, 2, 1], stock, lead,
        DeterministicRollingHorizonPolicy(item, [2, 2, 2, 2]),
        holding_cost_per_unit=1, stockout_cost_per_unit=penalty,
        order_cost_per_order=1, order_cost_per_unit=.5)


def test_joint_simulation_matches_independent_with_identical_deterministic_inputs():
    articles = {str(lead): article(lead, review=2) for lead in [0, 1, 2]}
    expected = simulate_replenishment_for_articles(articles)
    result = simulate_coordinated_replenishment(articles=articles,
        demand_scenarios={i: [[2]*4] for i in articles})
    for i in articles:
        assert result.simulations[i].snapshots == tuple(expected[i].snapshots)
        assert result.simulations[i].summary == expected[i].summary
        assert result.simulations[i].metadata.policy_family == 'coordinated_solver'
    assert len(result.plans) == 6
    assert len(result.resource_use) == 6


def test_joint_observations_are_post_demand_and_immutable():
    seen = []
    def provider(states):
        seen.append((states['A'].period, states['A'].on_hand))
        with pytest.raises(TypeError):
            states['A'] = None
        return {'A': [[2]*4]}
    simulate_coordinated_replenishment(articles={'A': article(stock=5)}, demand_scenarios=provider)
    assert seen[0] == (0, 4)
    assert [p for p, _ in seen] == list(range(6))


def test_shared_resources_are_checked_and_costs_match_execution():
    limits = SharedResourceLimits(max_order_units=3, procurement_budget=2.5,
                                  max_receiving_units=3)
    result = simulate_coordinated_replenishment(
        articles={'A': article(penalty=20), 'B': article(0, penalty=5)},
        demand_scenarios={'A': [[2]*4], 'B': [[2]*4]}, limits=limits)
    for use in result.resource_use:
        assert use.order_units <= 3
        assert use.procurement_cost <= 2.5
        assert use.receiving_units <= 3
    assert sum(u.procurement_cost for u in result.resource_use) == pytest.approx(
        sum(s.summary.ordering_cost for s in result.simulations.values()))
    for k, use in enumerate(result.resource_use):
        assert use.order_units == sum(s.snapshots[k].order_placed for s in result.simulations.values())
        assert use.receiving_units == sum(s.snapshots[k].received + (
            s.snapshots[k].order_placed if i == 'B' else 0)
            for i, s in result.simulations.items())


def test_joint_standard_rows_use_actual_snapshots_with_nontrivial_cadence():
    rows = generate_standard_simulation_rows(n_unique_ids=2, periods=6, seed=123,
        lead_time=0, holding_cost_per_unit=1, stockout_cost_per_unit=10,
        order_cost_per_order=1)
    ids = sorted({r.unique_id for r in rows})
    items = {i: SolverItemConfig(0, 20, review_period=2, holding_cost_per_unit=1,
        stockout_cost_per_unit=10, order_cost_per_order=1) for i in ids}
    policies = {i: ScenarioRollingHorizonPolicy(items[i], [[10, 20, 10]]) for i in ids}
    articles = build_article_configs_from_standard_rows(rows, policies=policies)
    result = simulate_coordinated_replenishment(articles=articles,
        demand_scenarios={i: [[10, 20, 10]] for i in ids},
        limits=SharedResourceLimits(max_order_units=20))
    decisions = build_replenishment_decisions_from_simulations(rows, result.simulations)
    for i in ids:
        for d, s in zip([d for d in decisions if d.unique_id == i], result.simulations[i].snapshots):
            assert d.ending_stock == s.ending_on_hand
            assert d.starting_stock == s.starting_on_hand
            assert d.missed_sales == s.lost_sales
            assert d.aggregation_window == 1


def test_joint_rejects_inconsistent_lengths_costs_leads_and_horizons():
    base = article()
    for bad, match in [
        (replace(base, periods=5), 'periods'),
        (replace(base, lead_time=2), 'lead_time'),
        (replace(base, order_cost_per_unit=5), 'cost'),
    ]:
        with pytest.raises(ValueError, match=match):
            simulate_coordinated_replenishment(articles={'A': base, 'B': bad},
                demand_scenarios={'A': [[2]*4], 'B': [[2]*4]})
    with pytest.raises(ValueError, match='horizon'):
        simulate_coordinated_replenishment(articles={'A': base}, demand_scenarios={'A': [[2]*3]})


@pytest.mark.parametrize('source_kind', ['callback', 'iterator'])
def test_live_demand_is_sampled_in_causal_order(source_kind):
    planned = []
    events = []
    def callback(period):
        assert len(planned) == period, 'Future actuals were requested before the current decision'
        events.append(('demand', period))
        return period % 3
    def stream():
        for period in range(6):
            yield callback(period)
    def scenarios(states):
        period = states['A'].period
        events.append(('plan', period))
        planned.append(period)
        return {'A': [[2]*4]}
    base = replace(article(), demand=callback if source_kind == 'callback' else stream())
    result = simulate_coordinated_replenishment(articles={'A': base}, demand_scenarios=scenarios)
    assert events == [entry for period in range(6) for entry in [('demand',period),('plan',period)]]
    assert [s.demand for s in result.simulations['A'].snapshots] == [0,1,2,0,1,2]
