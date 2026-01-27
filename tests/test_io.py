import csv

import pytest

from replenishment import (
    PointForecastRow,
    build_percentile_forecast_candidates,
    build_point_forecast_article_configs,
    iter_percentile_forecast_rows_from_csv,
    iter_point_forecast_rows_from_csv,
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
