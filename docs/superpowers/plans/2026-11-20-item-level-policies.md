# Item-Level Policies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add all six independent-item policies and six executable verification notebooks.

**Architecture:** Preserve the existing policy protocol and simulator timing. Add the six policies in a focused module, re-export from policies and package root, and add generic candidate tuning and standard-row config integration.

**Tech Stack:** Python >=3.10, standard library, existing pandas/matplotlib notebook dependencies, pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-item-level-policies-design.md`

## Global Constraints

- All decisions and tuning operate on one item at a time, without LP/MIP solvers or shared resource constraints.
- Add no mandatory dependencies.
- Preserve the user's existing untracked VN2 notebook.
- Keep existing event order and lost-sales behavior; do not add backlog mode.
- Review cadence is independent of aggregation.

## Review Focus

- Float/NaN/bool policy parameters must fail early rather than produce fractional orders.
- Externally constructed receipt schedules must be consistent and ordered.
- Stateful demand callables and generators must face identical traces across candidates.
- Lead-time zero must preserve lost demand and accurate decision-row stock reporting.
- Probabilistic model providers must not silently accept wrong horizon lengths or mutate stored samples.

### Task 1: Simulator foundations

**Files:** `src/replenishment/simulation.py`, `tests/test_policy_foundations.py`.
**Interfaces:** Add `InventoryState.arrival_schedule`, snapshot/summary `lost_sales`/`total_lost_sales` aliases, `SimulationMetadata.policy_name`/`policy_parameters`.

- [x] Write failing tests for L=0/1/2 hand traces, R>1, immutable dated observations and lost-sales aliases; check zero-lead snapshot starting stock.
- [x] Run `.venv/bin/python -m pytest tests/test_policy_foundations.py -q`; expect missing API/assertion failures.
- [x] Implement dated receipt observations, accurate snapshot starting stock, compatibility aliases, and optional metadata fields.
- [x] Run the foundation tests and existing suite; expect all pass.

### Task 2: Four static policies

**Files:** create `src/replenishment/item_policies.py`, `tests/test_item_policies.py`; modify `policies.py`, `__init__.py`.
**Interfaces:** `BaseStockPolicy(target_stock, review_period=1)`, `ConstantOrderPolicy(quantity, review_period=1)`, `CappedBaseStockPolicy(target_stock, cap, review_period=1)`, `sSPolicy(reorder_point, target_stock, review_period=1)`.

- [x] Write failing tests for exact rules, equality, pipeline subtraction, review cadence, nonfinite/fractional/bool/negative inputs, and cap equivalence.
- [x] Run `.venv/bin/python -m pytest tests/test_item_policies.py -q`; expect missing exports.
- [x] Implement policies and shared validation; export; fix `(r,Q)` docstring.
- [x] Run tests; expect all pass.

### Task 3: Probabilistic policies

**Files:** `item_policies.py`, `simulation.py`, `tests/test_probabilistic_policies.py`, exports.
**Interfaces:** `ProjectedInventoryLevelPolicy(target_stock, demand_pmf, lead_time, review_period=1)` with `projected_inventory(state)`; `ProtectionPeriodQuantilePolicy(quantile, lead_time, review_period=1, aggregate_samples=None, demand_paths=None)` with `target_for(state)` and `protection_period`. Sample inputs accept static iterables or state callbacks.

- [x] Write failing tests for exact PIL vs exhaustive enumeration, differing receipt timing, L=0/1, schedule validation, PMF validation, lead mismatch, empirical inverse CDF, H=max(1,L)+R-1, joint-path dependence, providers, generator freezing, and wrong shapes.
- [x] Run `.venv/bin/python -m pytest tests/test_probabilistic_policies.py -q`; expect missing API.
- [x] Implement exact discrete probability propagation and empirical quantile targeting; guard simulator lead mismatch for these policies.
- [x] Run tests; expect all pass.

### Task 4: Tuning, metadata, IO and aggregation boundaries

**Files:** `optimization.py`, `simulation.py`, `io.py`, `aggregation.py`, exports; create `tests/test_item_policy_integration.py`.
**Interfaces:** `PolicyOptimizationResult(label, policy, simulation, candidate_costs)`, `optimize_policy_candidates(*, candidates, periods, demand, initial_on_hand, lead_time, cost inputs)`, `build_article_configs_from_standard_rows(rows, *, policies, use_current_stock=None)`.

- [x] Write failing tests for generator/callable replay, stable ties, empty candidates, metadata retention, standard row stock selection and output, zero-lead decision stock correctness, and rejected unsupported aggregation.
- [x] Run `.venv/bin/python -m pytest tests/test_item_policy_integration.py -q`; expect missing API/assertion failures.
- [x] Implement generic per-item finite search, metadata propagation, row builder and decision-row stock correction, explicit aggregation errors.
- [x] Run the complete suite; expect all pass.

### Task 5: Examples and documentation

**Files:** six `notebooks/*_policy_example.ipynb` named in the spec; README, CHANGELOG, policy semantics documentation, example plot assets.
**Interfaces:** Notebook examples use the exported six policies, generic tuning and existing simulation/plot APIs.

- [x] Create six self-contained notebooks with worked assertions, chronological tuning/held-out evaluation, all required metrics and plots; include exact PIL reference and dependence/coverage quantile examples.
- [x] Execute all six notebooks with the project Python; expect no error outputs and all assertions pass. Save executed output and inspect plots.
- [x] Update README/changelog and document timing, lost-sales compatibility, local-vs-global scope and numerical/finite-search limitations.
- [x] Run full pytest and `git diff --check`; expect green suite and no whitespace errors.
- [x] Obtain one fresh whole-change code review, fix material findings with failing regression tests, then rerun affected notebooks and the full suite.

## Execution notes

User explicitly approved implementation and requested continuous execution. Work in the existing `feat/more_policies` checkout; leave changes reviewable and uncommitted unless requested. Track per-task evidence and deviations in the execution ledger. Do not add repeated approval gates or create an unrelated worktree.
