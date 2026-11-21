from dataclasses import replace
import pytest

pytest.importorskip('scipy')

from replenishment import (
    SolverItemConfig, SharedResourceLimits, InventoryState, plan_replenishment,
    ArticleSimulationConfig, DeterministicRollingHorizonPolicy,
    ScenarioRollingHorizonPolicy, simulate_coordinated_replenishment, RiskObjective,
)


def items():
    return {'A':SolverItemConfig(1,1,stockout_cost_per_unit=20,order_cost_per_unit=19),
            'B':SolverItemConfig(1,1,stockout_cost_per_unit=12,order_cost_per_unit=3)}


@pytest.mark.parametrize('mode,orders,revenue,profit,loss', [
    ('lost_sales',{'A':1,'B':0},20,1,12),
    ('total_cost',{'A':0,'B':1},12,9,23),
])
def test_revenue_and_profit_choose_different_budget_feasible_orders(mode,orders,revenue,profit,loss):
    configs=items()
    result=plan_replenishment(states={i:InventoryState(0,0,0,0) for i in configs},
        items=configs,demand_scenarios={i:[(1,)] for i in configs},
        limits=SharedResourceLimits(procurement_budget=19),cost_objective=mode)
    assert dict(result.first_orders)==orders
    assert result.cost_objective==mode
    realized_revenue=sum(c.stockout_cost_per_unit*(1-result.lost_sales[i][0][0])
                         for i,c in configs.items())
    purchases=sum(c.order_cost_per_unit*result.first_orders[i] for i,c in configs.items())
    assert purchases<=19
    assert realized_revenue==revenue
    assert realized_revenue-purchases==profit
    assert result.objective_value==pytest.approx(loss)


def test_revenue_objective_retains_fixed_order_costs_in_budget():
    configs={'A':SolverItemConfig(0,2,stockout_cost_per_unit=20,order_cost_per_unit=3,
                                 order_cost_per_order=5,holding_cost_per_unit=10)}
    result=plan_replenishment(states={'A':InventoryState(0,0,0,0)},items=configs,
        demand_scenarios={'A':[(2,)]},limits=SharedResourceLimits(procurement_budget=8),
        cost_objective='lost_sales')
    assert result.first_orders['A']==1
    assert result.objective_value==20  # Holding and setup costs excluded only from objective.


@pytest.mark.parametrize('cls,source', [(DeterministicRollingHorizonPolicy,[1]),
                                       (ScenarioRollingHorizonPolicy,[[1],[2]])])
def test_single_item_revenue_adapter_excludes_operating_costs(cls,source):
    config=SolverItemConfig(0,2,stockout_cost_per_unit=5,order_cost_per_unit=10,
                            holding_cost_per_unit=10)
    policy=cls(config,source,cost_objective='lost_sales')
    assert policy.plan(InventoryState(0,0,0,0)).first_orders['item']>=1
    assert cls(config,source).order_quantity_for(InventoryState(0,0,0,0))==0


def test_joint_revenue_run_preserves_actual_expenses_and_records_mode():
    configs=items()
    articles={i:ArticleSimulationConfig(periods=2,demand=[0,1],initial_on_hand=0,
        lead_time=c.lead_time,policy=DeterministicRollingHorizonPolicy(c,[1]),
        stockout_cost_per_unit=c.stockout_cost_per_unit,order_cost_per_unit=c.order_cost_per_unit)
        for i,c in configs.items()}
    result=simulate_coordinated_replenishment(articles=articles,
        demand_scenarios={i:[(1,)] for i in configs},
        limits=SharedResourceLimits(procurement_budget=19),cost_objective='lost_sales')
    assert result.resource_use[0].procurement_cost==19
    assert result.simulations['A'].summary.ordering_cost==38  # Final-period order also charged.
    assert dict(result.simulations['A'].metadata.policy_parameters)['cost_objective']=='lost_sales'
    assert result.plans[0].first_orders=={'A':1,'B':0}


def test_unknown_cost_objective_is_rejected():
    config=next(iter(items().values()))
    with pytest.raises(ValueError,match='cost_objective'):
        DeterministicRollingHorizonPolicy(config,[1],cost_objective='profit_typo')
    with pytest.raises(ValueError,match='cost_objective'):
        plan_replenishment(states={'A':InventoryState(0,0,0,0)},items={'A':config},
            demand_scenarios={'A':[(1,)]},cost_objective='profit_typo')
