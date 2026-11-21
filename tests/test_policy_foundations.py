from dataclasses import FrozenInstanceError

import pytest

from replenishment import InventoryState, ReorderPointPolicy, simulate_replenishment


class RecordingPolicy:
    def __init__(self):
        self.states = []

    def order_quantity_for(self, state):
        self.states.append(state)
        return 4 if state.period % 2 == 0 else 0


@pytest.mark.parametrize(
    'lead_time, receipts, endings, lost',
    [
        (0, [0, 0, 0, 0], [4, 1, 4, 2], 2),
        (1, [0, 4, 0, 4], [0, 1, 0, 2], 2),
        (2, [0, 0, 4, 0], [0, 0, 2, 0], 4),
    ],
)
def test_hand_calculated_receipt_demand_order_timing(lead_time, receipts, endings, lost):
    policy = RecordingPolicy()
    result = simulate_replenishment(
        periods=4, demand=[2, 3, 2, 2], initial_on_hand=1,
        lead_time=lead_time, policy=policy,
    )
    assert [s.received for s in result.snapshots] == receipts
    assert [s.ending_on_hand for s in result.snapshots] == endings
    assert [s.starting_on_hand for s in result.snapshots] == [
        1, endings[0] + receipts[1], endings[1] + receipts[2],
        endings[2] + receipts[3],
    ]
    assert result.summary.total_lost_sales == lost
    assert sum(s.lost_sales for s in result.snapshots) == lost
    assert result.summary.total_backorders == lost  # compatibility alias
    assert all(s.backorders == 0 for s in result.snapshots)
    assert all(s.backorders == 0 for s in policy.states)


def test_observation_arrival_schedule_is_dated_and_immutable():
    policy = RecordingPolicy()
    simulate_replenishment(
        periods=4, demand=[1, 1, 1, 1], initial_on_hand=0,
        lead_time=3, policy=policy,
    )
    assert [s.arrival_schedule for s in policy.states] == [
        (), ((3, 4),), ((3, 4),), ((5, 4),),
    ]
    for state in policy.states:
        assert state.on_order == sum(qty for _, qty in state.arrival_schedule)
    with pytest.raises(FrozenInstanceError):
        policy.states[1].arrival_schedule = ()
    assert InventoryState(0, 1, 0, 0).arrival_schedule == ()


def test_lost_demand_does_not_return_after_receipt():
    result = simulate_replenishment(
        periods=2, demand=[5, 0], initial_on_hand=0, lead_time=1,
        policy=ReorderPointPolicy(0, 3),
    )
    assert result.summary.total_lost_sales == 5
    assert result.snapshots[1].ending_on_hand == 3
    assert result.summary.total_fulfilled == 0
