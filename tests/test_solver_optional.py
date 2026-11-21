"""Core imports and existing simulations do not require the solver extra."""
import os
from pathlib import Path
import subprocess
import sys


def test_solver_import_and_core_execution_without_scipy():
    code = '''
import sys
class BlockScipy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'scipy' or fullname.startswith('scipy.'):
            raise ModuleNotFoundError('scipy deliberately unavailable')
sys.meta_path.insert(0, BlockScipy())
from replenishment import (BaseStockPolicy, simulate_replenishment,
    DeterministicRollingHorizonPolicy, SolverItemConfig, InventoryState)
r = simulate_replenishment(periods=2, demand=[1,2], initial_on_hand=0,
                           lead_time=1, policy=BaseStockPolicy(3))
assert r.summary.total_lost_sales == 1
policy = DeterministicRollingHorizonPolicy(SolverItemConfig(1, 3), [2])
try:
    policy.plan(InventoryState(0, 0, 0, 0))
except ModuleNotFoundError as exc:
    assert 'replenishment[solver]' in str(exc)
else:
    raise AssertionError('Expected an optional-dependency error')
assert 'scipy' not in sys.modules
'''
    env = dict(os.environ, MPLCONFIGDIR='/private/tmp/replenishment-mpl')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout+result.stderr
