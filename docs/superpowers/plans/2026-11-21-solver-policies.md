# Solver Policies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add inspectable HiGHS rolling-horizon policies and coordinated simulation with executed examples.

**Architecture:** A lazy SciPy MILP backend handles one or many items and finite scenario trees. Frozen policy adapters use the existing item protocol; a separate synchronous simulator executes joint decisions and returns existing result types.

**Tech Stack:** Python >=3.10, optional scipy>=1.11,<2 with HiGHS, pytest, existing pandas/matplotlib notebook workflow.

**Spec:** `docs/superpowers/specs/2026-11-21-solver-policies-design.md` (approved).

## Global Constraints

- Work on the user's `feat/solver_methods` branch in place; preserve the untracked VN2 notebook.
- Lazy optional solver dependency; no commercial solver; retain all existing APIs.
- Receipt -> demand -> order lost-sales timing, integer decisions, explicit finite horizon and zero terminal value.
- Nonanticipative scenario-tree recourse, equal scenario weights, and existing RiskObjective semantics.
- Shared per-period order, procurement and receiving limits apply in planning and execution.
- No commit/push/PR until requested; this overrides the generic skill's per-task commit convention.

## Review Focus

- Zero-lead orders carry current holding cost and cannot fulfill current lost demand (Task 1/3).
- Duplicate scenario prefixes must share future actions, including cross-item observations (Task 1).
- Fixed setup costs, packs and MOQ change feasible allocations (Task 1/3).
- Solver limit incumbents and near-integer vectors must not become unchecked actions (Task 1).
- Review cadence must not trigger legacy time aggregation in decision tables (Task 2/3).

### Task 1: Exact finite-scenario planner

**Files:** Create `src/replenishment/solver.py`, `tests/test_solver.py`; modify `pyproject.toml`.

**Interfaces:** Produce `SolverItemConfig`, `SharedResourceLimits`, `SolverOptions`, `PlanningError`, `RollingHorizonPlan`, `plan_replenishment(*, states, items, demand_scenarios, limits=..., objective=..., options=...)`. Inputs are matching string-keyed mappings; scenario paths start at t+1. Plan exposes frozen first orders, full scenario orders/inventory/loss/cost and numerical diagnostics.

- [x] Write tiny exhaustive reference tests for L=0/1/2, setups/packs/MOQ, dated receipts, cadence, scenario histories, all risk modes and resource allocation; validation and mocked solver statuses.
- [x] Run `.venv/bin/python -m pytest tests/test_solver.py -q`; expected missing-module failure before implementation.
- [x] Implement bounded sparse MILP assembly, lazy imports and independent replay checks; add optional solver extra and install it in local venv.
- [x] Run planner tests; expected all pass and exhaustive objectives agree.

### Task 2: Existing item workflow integration

**Files:** Create `src/replenishment/solver_policies.py`, `tests/test_solver_policies.py`; modify simulation, aggregation, IO, package exports.

**Interfaces:** Consume Task 1. Produce `DeterministicRollingHorizonPolicy(config, forecast, options=...)` and `ScenarioRollingHorizonPolicy(config, demand_scenarios, objective=..., options=...)`, both frozen, with `plan(state)` and `order_quantity_for(state)`, `lead_time`, `review_period`, `forecast_horizon`. Static sources copied, callbacks receive InventoryState.

- [x] Write failure-first tests for normal simulation/standard-row builder, immutable sources, callback period alignment, lead mismatch, metadata, cadence and aggregation rejection, imports without SciPy.
- [x] Run `.venv/bin/python -m pytest tests/test_solver_policies.py -q`; expected missing adapter/export failure.
- [x] Implement adapters and extend explicit built-in detection without creating import cycles. Decision rows use snapshot accounting for both solver families.
- [x] Run adapter and planner tests; expected all pass.

### Task 3: Joint synchronous simulation

**Files:** Create `src/replenishment/joint_simulation.py`, `tests/test_joint_simulation.py`; modify package exports.

**Interfaces:** Consume Tasks 1/2. Produce `simulate_coordinated_replenishment(*, articles, demand_scenarios, limits=..., objective=..., options=...) -> CoordinatedSimulationResult`, with existing ArticleSimulationConfig values, per-item results, plans and resource use. Item policies are solver adapters; plan costs and actual costs must agree on their item config.

- [x] Write failure-first tests for independent replay parity, mixed L=0/1/2, same-period observations, budget/receiving/order checks, mismatch rejection, snapshot/decision reconciliation and conflicting pipeline errors.
- [x] Run `.venv/bin/python -m pytest tests/test_joint_simulation.py -q`; expected missing-module failure.
- [x] Implement synchronous dynamics and execution validation using immutable provider observations and standard result types.
- [x] Run all tests; expected existing 312 plus new solver tests pass.

### Task 4: Executed examples and release documentation

**Files:** Create three `notebooks/{deterministic_rolling_horizon,scenario_rolling_horizon,coordinated_solver}_policy_example.ipynb`, `docs/solver_policies.md`, saved plots; modify README, CHANGELOG, CI.

**Interfaces:** Use Tasks 1–3 and existing standard dataset/config/decision/plot helpers.

- [x] Build three self-contained notebooks with generated three-item data, training-only forecasts/scenarios, held-out simulations, inspected plans, reconciled decision tables and saved item/aggregate plots. Coordinated example includes shared-use plot and feasible comparison.
- [x] Execute fresh IPython processes and save outputs; expected no cell errors and embedded PNGs equal saved assets.
- [x] Add formulation, limitations, install/API docs and optional solver CI job.
- [x] Run full suite and `git diff --check`; request independent whole-change code review, fix material findings with regression tests, then rerun affected checks.
