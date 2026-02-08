"""Replenishment simulation library."""

from importlib.metadata import PackageNotFoundError, version as _dist_version

try:
    from ._version import version as __version__
except ModuleNotFoundError:
    try:
        __version__ = _dist_version("replenishment")
    except PackageNotFoundError:
        __version__ = "0.0.0"

from .policies import (
    EmpiricalMultiplierPolicy,
    ForecastBasedPolicy,
    MeanForecastPolicy,
    ForecastSeriesPolicy,
    LeadTimeForecastOptimizationPolicy,
    PercentileForecastOptimizationPolicy,
    PointForecastOptimizationPolicy,
    RopEmpiricalMultiplierPolicy,
    RopPercentileForecastOptimizationPolicy,
    RopPointForecastOptimizationPolicy,
    ReorderPointPolicy,
)
from .aggregation import simulate_replenishment_with_aggregation
from .optimization import (
    AggregationForecastTargetOptimizationResult,
    AggregationServiceLevelOptimizationResult,
    AggregationWindowOptimizationResult,
    EmpiricalCalibrationConfig,
    EmpiricalCalibrationResult,
    ForecastCandidatesConfig,
    ForecastTargetOptimizationResult,
    PercentileForecastOptimizationResult,
    PointForecastOptimizationResult,
    ServiceLevelOptimizationResult,
    calibrate_empirical_multipliers,
    evaluate_aggregation_and_forecast_target_costs,
    evaluate_aggregation_and_service_level_factor_costs,
    evaluate_empirical_multiplier_lost_sales,
    evaluate_forecast_target_costs,
    evaluate_service_level_factor_costs,
    optimize_aggregation_and_forecast_targets,
    optimize_aggregation_and_service_level_factors,
    optimize_aggregation_windows,
    optimize_forecast_targets,
    optimize_service_level_factors,
    percentile_forecast_optimisation,
    point_forecast_optimisation,
)
from .io import (
    PercentileForecastRow,
    PointForecastRow,
    StandardSimulationRow,
    build_percentile_forecast_candidates,
    build_percentile_forecast_candidates_from_standard_rows,
    build_lead_time_forecast_article_configs_from_standard_rows,
    build_point_forecast_article_configs,
    build_point_forecast_article_configs_from_standard_rows,
    compute_backtest_rmse_by_article,
    optimize_point_forecast_policy_and_simulate_actuals,
    build_replenishment_decisions_from_optimization_results,
    build_replenishment_decisions_from_simulations,
    generate_standard_simulation_rows,
    iter_percentile_forecast_rows_from_csv,
    iter_point_forecast_rows_from_csv,
    iter_standard_simulation_rows_from_csv,
    replenishment_decision_rows_to_dataframe,
    replenishment_decision_rows_to_dicts,
    ReplenishmentDecisionRow,
    split_standard_simulation_rows,
    standard_simulation_rows_to_dataframe,
    standard_simulation_rows_from_dataframe,
    standard_simulation_rows_to_dicts,
    write_standard_simulation_rows_to_csv,
)
from .simulation import (
    ArticleSimulationConfig,
    DemandModel,
    InventoryState,
    SimulationMetadata,
    SimulationResult,
    SimulationSummary,
    simulate_replenishment,
    simulate_replenishment_for_articles,
)
try:
    from .plotting import plot_replenishment_decisions
    _HAS_PLOTTING = True
except ModuleNotFoundError:
    plot_replenishment_decisions = None
    _HAS_PLOTTING = False

__all__ = [
    "__version__",
    "DemandModel",
    "InventoryState",
    "ArticleSimulationConfig",
    "EmpiricalMultiplierPolicy",
    "ForecastBasedPolicy",
    "MeanForecastPolicy",
    "ForecastSeriesPolicy",
    "LeadTimeForecastOptimizationPolicy",
    "PercentileForecastOptimizationPolicy",
    "PointForecastOptimizationPolicy",
    "RopEmpiricalMultiplierPolicy",
    "RopPercentileForecastOptimizationPolicy",
    "RopPointForecastOptimizationPolicy",
    "ReorderPointPolicy",
    "SimulationMetadata",
    "SimulationResult",
    "SimulationSummary",
    "EmpiricalCalibrationConfig",
    "EmpiricalCalibrationResult",
    "ForecastCandidatesConfig",
    "ForecastTargetOptimizationResult",
    "PercentileForecastOptimizationResult",
    "AggregationForecastTargetOptimizationResult",
    "AggregationWindowOptimizationResult",
    "AggregationServiceLevelOptimizationResult",
    "ServiceLevelOptimizationResult",
    "PointForecastOptimizationResult",
    "PointForecastRow",
    "PercentileForecastRow",
    "StandardSimulationRow",
    "ReplenishmentDecisionRow",
    "generate_standard_simulation_rows",
    "iter_point_forecast_rows_from_csv",
    "iter_percentile_forecast_rows_from_csv",
    "iter_standard_simulation_rows_from_csv",
    "split_standard_simulation_rows",
    "standard_simulation_rows_to_dataframe",
    "standard_simulation_rows_from_dataframe",
    "standard_simulation_rows_to_dicts",
    "replenishment_decision_rows_to_dataframe",
    "replenishment_decision_rows_to_dicts",
    "write_standard_simulation_rows_to_csv",
    "build_point_forecast_article_configs",
    "build_percentile_forecast_candidates",
    "build_point_forecast_article_configs_from_standard_rows",
    "optimize_point_forecast_policy_and_simulate_actuals",
    "build_percentile_forecast_candidates_from_standard_rows",
    "build_lead_time_forecast_article_configs_from_standard_rows",
    "compute_backtest_rmse_by_article",
    "build_replenishment_decisions_from_simulations",
    "build_replenishment_decisions_from_optimization_results",
    "simulate_replenishment",
    "simulate_replenishment_with_aggregation",
    "simulate_replenishment_for_articles",
    "optimize_aggregation_and_forecast_targets",
    "optimize_aggregation_and_service_level_factors",
    "optimize_aggregation_windows",
    "evaluate_forecast_target_costs",
    "evaluate_service_level_factor_costs",
    "evaluate_aggregation_and_forecast_target_costs",
    "evaluate_aggregation_and_service_level_factor_costs",
    "optimize_forecast_targets",
    "optimize_service_level_factors",
    "calibrate_empirical_multipliers",
    "evaluate_empirical_multiplier_lost_sales",
    "percentile_forecast_optimisation",
    "point_forecast_optimisation",
]

if _HAS_PLOTTING:
    __all__.append("plot_replenishment_decisions")
