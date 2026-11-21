# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Optional PyTorch `advanced` extra, structured neural residual policy with
  supervised training, pure-Python deployment and numeric JSON checkpoints.
- Single-item HDPO with differentiable lost-sales rollouts, separate exact
  integer validation, checkpoint selection and explicit rounding diagnostics.
- Solver-free mean/CVaR/blended/worst-scenario candidate search and matching
  differentiable risk objectives, including fractional CVaR tails.
- Three executed advanced-method notebooks and CPU learning tests in CI.
- Standardized all nine new-method notebooks on generated standard rows/DataFrames,
  per-article policy inspection, simulation/decision tables and the existing
  individual/aggregate stock and missed-sales plotting workflow.

- Six independent-item, solver-free policies: static base-stock, constant-order,
  capped base-stock, independent `(s,S)`, projected inventory-level and
  protection-period quantile targeting.
- A separate executed verification/tuning notebook for each policy, with generated
  multi-item data, held-out outcomes, metrics and shared replenishment plots.
- Immutable dated receipt observations, explicit lost-sales metric aliases,
  per-item candidate-policy cost search, generic standard-row config building,
  and policy metadata.
- Documented event timing, model assumptions and boundaries for future shared
  constraint, solver and research PRs.

### Fixed

- Zero-lead-time snapshots now report stock available before current demand,
  excluding the order placed afterward; demand/order trajectories are unchanged.
- New-policy decision rows report actual simulated stock/lost sales; legacy
  reconstructed stock paths now include immediate orders after demand.
- Corrected the existing `(r,Q)` policy docstring to describe fixed quantities.

### Notes

- Existing public backlog-named fields remain for compatibility. Simulation is
  still lost sales; a backlog mode has not been added.
- New policies reject implicit time aggregation beyond an identity window;
  explicit review cadence remains supported without aggregating demand.

## [0.1.2] - 2026-03-17

### Added

- Added `k*RMSE` as an explicit safety stock method.
- Added demand-scaled forecast buffering options for point-forecast policies.
- Added a notebook example showing how forecast-level buffering increases safety stock on replenishment timelines.
- Added top-level contribution guidance in `CONTRIBUTING.md`.
- Added an MIT license.

### Changed

- Improved fill-rate safety stock handling to use protection-window demand.
- Preserved lead-time offset behavior in the empirical multiplier policy.
- Hardened the release workflow for reruns.
- Refreshed the README, installation instructions, supported Python versions, and checked-in plot assets.
- Expanded CI coverage to Python 3.10, 3.11, 3.12, and 3.13.

## [0.1.1] - 2025-02-08

### Added

- Added the percentile optimization step-by-step workflow and release automation improvements.

## [0.1.0] - 2025-02-07

### Added

- Initial PyPI release setup.
