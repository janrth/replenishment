# Standard New-Method Notebook Workflows

> **For agentic workers:** Use superpowers:executing-plans for continuous inline execution; the user has delegated implementation judgment.

**Goal:** Make all nine new-method examples follow the existing generated-standard-data, split, fit, simulate, inspect and plot workflow.
**Architecture:** Self-contained notebooks reuse public dataset/config/decision/plot APIs. Method-specific tuning/training is the only changing middle section. Keep focused mathematical verification in appendices; replace bespoke primary demonstrations with standard tables and charts.
**Tech stack:** Existing Python/pandas/matplotlib APIs; optional Torch for the two learning notebooks.

## Constraints and review focus

- Preserve the neural notebook's direct missed-sales objective; do not revert to teacher imitation.
- Independently fit each article; no shared stock/allocation or LP/MIP solver.
- Generate the same seeded three-item standard dataset in every example, with explicit seasonal, intermittent and rising profiles and evaluation actuals equal to simulated demand.
- Split chronologically into training/validation/evaluation; no evaluation actuals in model fitting.
- Train distributions/bootstrap samples only on training rows. Treat future forecast vectors as explicitly supplied exogenous inputs.
- Use `build_article_configs_from_standard_rows`, `simulate_replenishment_for_articles`, `build_replenishment_decisions_from_simulations`, and `plot_replenishment_decisions` for individual IDs and aggregate totals.
- Aggregate missed sales after per-item simulation. Assert decision tables, summaries, and cumulative plotted loss/share agree.
- Preserve unrelated/legacy notebooks and existing library APIs. Do not commit/push.

## Tasks

- [x] Inspect legacy notebook conventions and shared plotting semantics.
- [x] Rewrite the six baseline/probabilistic notebooks with the standard workflow and retain their worked reference assertions.
- [x] Apply the same format to neural/HDPO/risk notebooks, preserving separate validation, objective diagnostics, per-item policy inspection and scenario assumptions.
- [x] Execute all nine, validate stock/demand/decision/plot reconciliation and inspect representative generated charts; run the existing suite if integration problems warrant fixes.
- [x] Update documentation, obtain an independent notebook/workflow review and address material findings; finish with verified saved outputs and whitespace checks.
