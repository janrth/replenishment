# Advanced Item Policies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an optional structured neural policy, HDPO training and solver-free scenario-risk optimisation, with three executed notebooks.
**Architecture:** Pure-Python risk objectives/candidate search; optional Torch neural architecture and continuous differentiable trainer; frozen integer deployment adapter integrated with current simulation.
**Tech Stack:** Python >=3.10; existing dependencies; optional torch>=2.6,<3; pytest.
**Spec:** `docs/superpowers/specs/2026-11-20-advanced-item-policies-design.md`.

## Global Constraints

- Independent items, no LP/MIP solver or shared constraints.
- Torch is optional; core/risk imports work without it.
- Preserve six existing policies, their tests/notebooks, and unrelated VN2 notebook.
- Training decisions never use future realized demand; validation is separate from gradients.
- Continuous HDPO training rejects positive fixed ordering costs; exact integer evaluation reports rounding gap.
- User opted for implementation using judgment; execute inline without further approval gates.

## Review Focus

- Missing Torch must not break lightweight installs or core star imports.
- Model/forecast shapes and finite numeric values must fail early, with no mutable weight sharing.
- Tensor receipt/review timing must agree with Python, especially L=0.
- Empirical CVaR must handle fractional tails and exact confidence boundaries.
- Training must update parameters, preserve caller RNG, and select checkpoints only using explicit validation.

### Task 1: Risk objectives and scenario candidate search

Files: create `risk.py`, `tests/test_risk.py`; root exports.
Interfaces: `RiskObjective(mode='mean', alpha=.9, risk_weight=1)`, `empirical_cvar(costs, alpha=.9)`, `optimize_risk_sensitive_policy_candidates(*, candidates, demand_scenarios, initial_on_hand, lead_time, objective, costs)` → immutable result.
- [x] Write/run RED tests for fractional CVaR, tail boundaries, risk-vs-mean choices, stable ties, scenario generator replay and invalid shapes/values.
- [x] Implement dependency-free objectives, fixed-horizon materialization and candidate evaluation.
- [x] Run full suite; record evidence.

### Task 2: Optional neural architecture, inference and supervised training

Files: create `neural.py`, `tests/test_neural.py`; pyproject optional extra; lazy root exports; integrate simulator/item metadata/aggregation/IO.
Interfaces: `StructuredNeuralPolicy`, `create_structured_neural_policy`, `train_structured_neural_policy`, JSON `save`/`load`.
- [x] Write/run RED tests for no-Torch import isolation, zero-residual baseline, pipeline/as-of forecast features, validation, defensive copies, checkpoint roundtrip and supervised learning.
- [x] Implement residual MLP, immutable deployment adapter, safe numeric JSON and supervised trainer; register built-in behavior without core Torch imports.
- [x] Run core and neural suites, record evidence.

### Task 3: Differentiable rollout and HDPO

Files: create `hdpo.py`, `tests/test_hdpo.py`; lazy exports; advanced CI job.
Interfaces: tensor rollout with scenario costs/trajectories; differentiable risk scoring; `train_hdpo` returning deployed policy/history/selected checkpoint.
- [x] Write/run RED tests for exact integer-action parity, independent gradient finite differences, risk gradients, updates/reproducibility/RNG scope, fixed-cost rejection, validation selection and shapes.
- [x] Implement continuous tensor lost-sales rollout, Adam training with local seeding, validation-driven integer checkpoint selection and mean/CVaR/worst objectives.
- [x] Run full suite and core-without-Torch import check; record evidence.

### Task 4: Three research examples and documentation

Files: three named notebooks, plots, `docs/advanced_item_policies.md`, README, changelog.
- [x] Create examples with hand assertions, learning/tuning, held-out baselines, risk and rounding diagnostics, seeds/budgets/runtime.
- [x] Execute all three notebooks with actual Torch runtime, persist outputs, inspect plots.
- [x] Run all tests, existing six notebooks as regression when integration changes warrant, compile and whitespace checks.
- [x] Obtain one independent whole-change review; reproduce/fix material findings with RED→GREEN tests; final full verification.

Leave changes reviewable and uncommitted on existing `feat/more_policies` branch.
