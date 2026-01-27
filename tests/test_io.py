import csv

import pytest

from replenishment import (
    PointForecastRow,
    StandardSimulationRow,
    build_percentile_forecast_candidates,
    build_percentile_forecast_candidates_from_standard_rows,
    build_point_forecast_article_configs,
    build_point_forecast_article_configs_from_standard_rows,
    iter_percentile_forecast_rows_from_csv,
    iter_point_forecast_rows_from_csv,
    iter_standard_simulation_rows_from_csv,
    standard_simulation_rows_from_dataframe,
)


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_build_point_forecast_article_configs_from_csv(tmp_path):
    path = tmp_path / "point.csv"
    _write_csv(
        path,
        ["unique_id", "period", "demand", "forecast", "actual"],
        [
            ["A", 0, 10, 12, 11],
            ["A", 1, 9, 11, 10],
            ["B", 0, 3, 4, 5],
            ["B", 1, 4, 5, 4],
        ],
    )

    rows = iter_point_forecast_rows_from_csv(str(path))
    configs = build_point_forecast_article_configs(
        rows,
        lead_time={"A": 1, "B": 2},
        initial_on_hand=5,
        service_level_factor=0.8,
    )

    assert set(configs.keys()) == {"A", "B"}
    assert configs["A"].periods == 2
    assert configs["A"].lead_time == 1
    assert configs["B"].lead_time == 2


def test_build_percentile_forecast_candidates_from_csv(tmp_path):
    path = tmp_path / "percentile.csv"
    _write_csv(
        path,
        ["unique_id", "period", "demand", "target", "forecast"],
        [
            ["A", 0, 10, "p50", 11],
            ["A", 0, 10, "p90", 14],
            ["A", 1, 8, "p50", 9],
            ["A", 1, 8, "p90", 12],
        ],
    )

    rows = iter_percentile_forecast_rows_from_csv(str(path))
    configs = build_percentile_forecast_candidates(
        rows,
        lead_time=1,
        initial_on_hand=5,
    )

    assert set(configs.keys()) == {"A"}
    config = configs["A"]
    assert config.periods == 2
    assert set(config.forecast_candidates.keys()) == {"p50", "p90"}
    assert config.forecast_candidates["p50"] == [11, 9]


def test_build_point_forecast_article_configs_requires_contiguous_periods():
    rows = [PointForecastRow("A", 1, 10, 11, 12)]

    with pytest.raises(ValueError, match="Missing periods"):
        build_point_forecast_article_configs(
            rows,
            lead_time=1,
            initial_on_hand=5,
            service_level_factor=0.9,
        )


def test_iter_standard_simulation_rows_from_csv(tmp_path):
    path = tmp_path / "standard.csv"
    _write_csv(
        path,
        [
            "unique_id",
            "ds",
            "demand",
            "forecast",
            "actuals",
            "holding_cost_per_unit",
            "stockout_cost_per_unit",
            "order_cost_per_order",
            "lead_time",
            "initial_on_hand",
            "current_stock",
            "is_forecast",
            "forecast_p50",
            "forecast_p90",
        ],
        [
            ["A", "2024-01-01", 10, 12, 11, 0.5, 3.0, 2.0, 1, 5, 8, "true", 11, 14],
        ],
    )

    rows = list(iter_standard_simulation_rows_from_csv(str(path)))

    assert rows == [
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-01",
            demand=10,
            forecast=12,
            actuals=11,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=5,
            current_stock=8,
            forecast_percentiles={"p50": 11, "p90": 14},
            is_forecast=True,
        )
    ]


def test_build_configs_from_standard_rows():
    rows = [
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-01",
            demand=10,
            forecast=12,
            actuals=11,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=5,
            current_stock=8,
            forecast_percentiles={"p50": 11, "p90": 14},
        ),
        StandardSimulationRow(
            unique_id="A",
            ds="2024-01-02",
            demand=9,
            forecast=10,
            actuals=9,
            holding_cost_per_unit=0.5,
            stockout_cost_per_unit=3.0,
            order_cost_per_order=2.0,
            lead_time=1,
            initial_on_hand=5,
            current_stock=8,
            forecast_percentiles={"p50": 10, "p90": 13},
        ),
    ]

    point_configs = build_point_forecast_article_configs_from_standard_rows(
        rows,
        service_level_factor=0.9,
    )
    percentile_configs = build_percentile_forecast_candidates_from_standard_rows(rows)

    assert set(point_configs.keys()) == {"A"}
    assert point_configs["A"].periods == 2
    assert point_configs["A"].initial_on_hand == 5
    assert set(percentile_configs["A"].forecast_candidates.keys()) == {"p50", "p90"}


def test_standard_simulation_rows_from_dataframe_history_and_cutoff():
    class FakeDataFrame:
        def __init__(self, records):
            self._records = records

        def to_dict(self, orient="records"):
            assert orient == "records"
            return self._records

    records = [
        {
            "unique_id": "A",
            "ds": "2024-01-01",
            "history": 10,
            "forecast": 12,
            "holding_cost_per_unit": 0.5,
            "stockout_cost_per_unit": 3.0,
            "order_cost_per_order": 2.0,
            "lead_time": 1,
            "initial_on_hand": 5,
            "current_stock": 8,
            "forecast_p50": 11,
        },
        {
            "unique_id": "A",
            "ds": "2024-02-01",
            "history": float("nan"),
            "forecast": 13,
            "holding_cost_per_unit": 0.5,
            "stockout_cost_per_unit": 3.0,
            "order_cost_per_order": 2.0,
            "lead_time": 1,
            "initial_on_hand": 5,
            "current_stock": 8,
            "forecast_p50": 12,
        },
    ]

    rows = standard_simulation_rows_from_dataframe(
        FakeDataFrame(records),
        cutoff="2024-01-01",
    )

    assert rows[0].actuals == 10
    assert rows[0].is_forecast is False
    assert rows[1].actuals is None
    assert rows[1].is_forecast is True


def test_iter_standard_simulation_rows_warns_on_missing_columns(tmp_path):
    path = tmp_path / "missing.csv"
    _write_csv(
        path,
        ["unique_id", "ds", "demand"],
        [
            ["A", "2024-01-01", 10],
        ],
    )

    with pytest.warns(UserWarning, match="Missing required columns"):
        with pytest.raises(ValueError, match="missing required columns"):
            list(iter_standard_simulation_rows_from_csv(str(path)))


def test_iter_standard_simulation_rows_requires_current_stock(tmp_path):
    path = tmp_path / "missing_stock.csv"
    _write_csv(
        path,
        [
            "unique_id",
            "ds",
            "demand",
            "forecast",
            "actuals",
            "holding_cost_per_unit",
            "stockout_cost_per_unit",
            "order_cost_per_order",
            "lead_time",
            "initial_on_hand",
        ],
        [
            ["A", "2024-01-01", 10, 12, 11, 0.5, 3.0, 2.0, 1, 5],
        ],
    )

    with pytest.warns(UserWarning, match="current_stock"):
        with pytest.raises(ValueError, match="current_stock"):
            list(iter_standard_simulation_rows_from_csv(str(path)))


def test_iter_standard_simulation_rows_requires_matching_initial_inventory(tmp_path):
    path = tmp_path / "mismatch.csv"
    _write_csv(
        path,
        [
            "unique_id",
            "ds",
            "demand",
            "forecast",
            "actuals",
            "holding_cost_per_unit",
            "stockout_cost_per_unit",
            "order_cost_per_order",
            "lead_time",
            "initial_on_hand",
            "initial_demand",
            "current_stock",
        ],
        [
            ["A", "2024-01-01", 10, 12, 11, 0.5, 3.0, 2.0, 1, 5, 7, 8],
        ],
    )

    with pytest.warns(UserWarning, match="Initial inventory mismatch"):
        with pytest.raises(ValueError, match="initial_on_hand and initial_demand"):
            list(iter_standard_simulation_rows_from_csv(str(path)))
