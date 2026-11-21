import pytest

from replenishment import (
    BaseStockPolicy, ConstantOrderPolicy, CappedBaseStockPolicy, sSPolicy,
    InventoryState, simulate_replenishment,
)


def state(stock=0, pipeline=0, period=0):
    return InventoryState(period, stock, pipeline, 0)


def test_base_stock_counts_pipeline_once_and_stops_at_target():
    policy = BaseStockPolicy(10)
    assert policy.order_quantity_for(state(2, 3)) == 5
    assert policy.order_quantity_for(state(10)) == 0
    assert policy.order_quantity_for(state(20)) == 0


def test_constant_order_ignores_inventory_position():
    policy = ConstantOrderPolicy(4)
    assert [policy.order_quantity_for(s) for s in [state(), state(500, 200)]] == [4, 4]


def test_capped_base_stock_limits_orders_and_large_cap_matches_uncapped():
    assert CappedBaseStockPolicy(10, 3).order_quantity_for(state(2)) == 3
    assert CappedBaseStockPolicy(10, 0).order_quantity_for(state()) == 0
    for stock in [0, 5, 10, 15]:
        assert CappedBaseStockPolicy(10, 100).order_quantity_for(state(stock, 2)) == (
            BaseStockPolicy(10).order_quantity_for(state(stock, 2))
        )


def test_s_s_equality_and_independent_threshold():
    policy = sSPolicy(4, 10)
    assert policy.order_quantity_for(state(2, 2)) == 6
    assert policy.order_quantity_for(state(5)) == 0
    assert policy.order_quantity_for(state(1)) == 9
    assert sSPolicy(-1, 10).order_quantity_for(state()) == 0
    assert sSPolicy(4, 4).order_quantity_for(state(4)) == 0


@pytest.mark.parametrize('policy', [
    BaseStockPolicy(10, review_period=3), ConstantOrderPolicy(4, review_period=3),
    CappedBaseStockPolicy(10, 4, review_period=3), sSPolicy(4, 10, review_period=3),
])
def test_review_cadence_is_not_demand_aggregation(policy):
    result = simulate_replenishment(
        periods=5, demand=[1, 2, 3, 4, 5], initial_on_hand=0, lead_time=1,
        policy=policy,
    )
    assert result.snapshots[0].order_placed > 0
    assert [result.snapshots[i].order_placed for i in [1, 2, 4]] == [0, 0, 0]
    assert [s.demand for s in result.snapshots] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize('bad', [-1, 1.5, float('nan'), float('inf'), True, '4'])
@pytest.mark.parametrize('factory', [BaseStockPolicy, ConstantOrderPolicy,
    lambda x: CappedBaseStockPolicy(10, x), lambda x: CappedBaseStockPolicy(x, 2),
    lambda x: sSPolicy(0, x)])
def test_invalid_quantities_fail_at_construction(factory, bad):
    with pytest.raises((ValueError, TypeError)):
        factory(bad)


@pytest.mark.parametrize('factory', [BaseStockPolicy, ConstantOrderPolicy,
    lambda x, **kw: CappedBaseStockPolicy(x, 2, **kw), lambda x, **kw: sSPolicy(0, x, **kw)])
@pytest.mark.parametrize('bad', [0, -1, 1.5, True])
def test_invalid_review_cadence(factory, bad):
    with pytest.raises((ValueError, TypeError)):
        factory(5, review_period=bad)


def test_s_s_rejects_inverted_thresholds_and_noninteger_threshold():
    with pytest.raises(ValueError):
        sSPolicy(5, 4)
    with pytest.raises(TypeError):
        sSPolicy(1.5, 10)
