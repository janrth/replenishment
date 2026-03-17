# Changelog

All notable changes to this project will be documented in this file.

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
