"""Plotting helpers for replenishment simulations."""

from __future__ import annotations

from collections.abc import Iterable
import warnings
import math
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from .aggregation import aggregate_lead_time, aggregate_series
from .service_levels import service_level_multiplier
from .io import (
    ReplenishmentDecisionRow,
    StandardSimulationRow,
    replenishment_decision_rows_to_dataframe,
    standard_simulation_rows_to_dataframe,
)


def _normalize_rows(
    rows: Iterable[StandardSimulationRow] | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    return standard_simulation_rows_to_dataframe(rows, library="pandas")


def _normalize_decisions(
    decisions: Iterable[ReplenishmentDecisionRow] | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(decisions, pd.DataFrame):
        return decisions.copy()
    return replenishment_decision_rows_to_dataframe(decisions, library="pandas")


def _prepare_timeseries(
    rows: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    unique_id: str | None,
    aggregate: bool,
) -> pd.DataFrame:
    if aggregate and unique_id is not None:
        raise ValueError("Provide either aggregate=True or a unique_id, not both.")
    if not aggregate and unique_id is None:
        raise ValueError("unique_id is required when aggregate=False.")

    rows = rows.copy()
    decisions = decisions.copy()

    rows["ds"] = pd.to_datetime(rows["ds"])
    decisions["ds"] = pd.to_datetime(decisions["ds"])

    if aggregate:
        grouped_rows = (
            rows.groupby("ds", as_index=False)
            .agg(
                actuals=("actuals", lambda values: values.sum(min_count=1)),
                forecast=("forecast", lambda values: values.sum(min_count=1)),
            )
            .sort_values("ds")
        )
        agg_spec: dict[str, tuple[str, str]] = {
            "replenishment": ("quantity", "sum"),
        }
        if "incoming_stock" in decisions.columns:
            agg_spec["incoming_stock"] = ("incoming_stock", "sum")
        grouped_decisions = (
            decisions.groupby("ds", as_index=False)
            .agg(**agg_spec)
            .sort_values("ds")
        )
    else:
        grouped_rows = rows.loc[
            rows["unique_id"] == unique_id, ["ds", "actuals", "forecast"]
        ]
        cols = ["ds", "quantity"]
        if "incoming_stock" in decisions.columns:
            cols.append("incoming_stock")
        grouped_decisions = decisions.loc[
            decisions["unique_id"] == unique_id, cols
        ].rename(columns={"quantity": "replenishment"})

    data = grouped_rows.merge(grouped_decisions, on="ds", how="outer")
    data["replenishment"] = data["replenishment"].fillna(0)
    if "incoming_stock" in data.columns:
        data["incoming_stock"] = data["incoming_stock"].fillna(0)
    return data.sort_values("ds")


def _initial_stock_level(
    rows: pd.DataFrame,
    *,
    unique_id: str | None,
    aggregate: bool,
    start_date: pd.Timestamp | None,
) -> float:
    if aggregate:
        total = 0.0
        for article_id, group in rows.groupby("unique_id"):
            total += _initial_stock_level(
                group, unique_id=article_id, aggregate=False, start_date=start_date
            )
        return total
    if unique_id is None:
        return 0.0
    subset = rows.loc[rows["unique_id"] == unique_id].sort_values("ds")
    if start_date is not None:
        history = subset.loc[subset["ds"] < start_date]
    else:
        history = subset
    if history.empty:
        history = subset
    if history.empty:
        return 0.0
    current = history["current_stock"].iloc[-1]
    if pd.isna(current):
        current = history["initial_on_hand"].iloc[-1]
    return float(current)


def _build_stock_series(
    series: pd.DataFrame,
    initial_stock: float,
    *,
    start_date: pd.Timestamp | None,
) -> tuple[pd.Series, pd.Series]:
    demand = series["forecast"].where(series["forecast"].notna(), series["actuals"])
    demand = demand.fillna(0).astype(float)
    if "incoming_stock" in series.columns:
        replenishment = series["incoming_stock"].fillna(0).astype(float)
    else:
        replenishment = series["replenishment"].fillna(0).astype(float)
    stock_values: list[float] = []
    lost_sales: list[float] = []
    stock = float(initial_stock)
    for ds, replenishment_qty, consumption in zip(
        series["ds"], replenishment, demand, strict=False
    ):
        if start_date is not None and ds < start_date:
            stock_values.append(float("nan"))
            lost_sales.append(float("nan"))
            continue
        available = stock + float(replenishment_qty)
        stock_after = available - float(consumption)
        if stock_after < 0:
            lost_sales.append(-stock_after)
            stock = 0.0
        else:
            lost_sales.append(0.0)
            stock = stock_after
        stock_values.append(stock)
    return pd.Series(stock_values, index=series.index), pd.Series(
        lost_sales, index=series.index
    )


def _trim_actuals(values: pd.Series) -> list[float]:
    trimmed: list[float] = []
    for value in values:
        if pd.isna(value):
            break
        trimmed.append(float(value))
    return trimmed


def _safety_stock_for_decisions(
    rows_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
    *,
    unique_id: str,
) -> pd.DataFrame | None:
    if decisions_df.empty:
        return None
    if "safety_stock" in decisions_df.columns:
        safety_series = pd.to_numeric(
            decisions_df["safety_stock"], errors="coerce"
        ).dropna()
        if not safety_series.empty:
            window = 1
            if "aggregation_window" in decisions_df.columns:
                window_values = pd.to_numeric(
                    decisions_df["aggregation_window"], errors="coerce"
                ).dropna()
                if not window_values.empty:
                    window = int(window_values.iloc[0])
            safety_values = decisions_df["safety_stock"].astype(float).tolist()
            safety_per_period = [
                value / window if window > 0 else value for value in safety_values
            ]
            return pd.DataFrame(
                {
                    "ds": decisions_df["ds"].values,
                    "safety_stock": safety_values,
                    "safety_stock_per_period": safety_per_period,
                    "aggregation_window": window,
                }
            )
    sigma_values = pd.to_numeric(decisions_df["sigma"], errors="coerce")
    sigma_values = sigma_values.dropna()
    if sigma_values.empty:
        return None
    sigma_value = float(sigma_values.iloc[0])
    mode_value = None
    if "service_level_mode" in decisions_df.columns:
        mode_values = decisions_df["service_level_mode"].dropna()
        if not mode_values.empty:
            mode_value = str(mode_values.iloc[0])
    sigma_multiplier = service_level_multiplier(sigma_value, mode_value)

    window = 1
    if "aggregation_window" in decisions_df.columns:
        window_values = pd.to_numeric(
            decisions_df["aggregation_window"], errors="coerce"
        ).dropna()
        if not window_values.empty:
            window = int(window_values.iloc[0])

    lead_time_values = pd.to_numeric(
        rows_df["lead_time"], errors="coerce"
    ).dropna()
    if lead_time_values.empty:
        return None
    lead_time = int(lead_time_values.iloc[0])
    if window > 1:
        lead_time = aggregate_lead_time(lead_time, window)

    ordered_rows = rows_df.sort_values("ds")
    forecast_values = ordered_rows["forecast"].astype(float).tolist()
    actuals_values = _trim_actuals(ordered_rows["actuals"])
    if window > 1:
        forecast_values = aggregate_series(
            forecast_values,
            periods=len(forecast_values),
            window=window,
            extend_last=True,
        )
        if actuals_values:
            actuals_values = aggregate_series(
                actuals_values,
                periods=len(actuals_values),
                window=window,
                extend_last=False,
            )

    decisions_df = decisions_df.sort_values("ds")
    safety_values: list[float] = []
    for period in range(len(decisions_df)):
        max_index = min(period, len(actuals_values), len(forecast_values))
        if max_index <= 0:
            rmse = 0.0
        else:
            errors = [
                actuals_values[index] - forecast_values[index]
                for index in range(max_index)
            ]
            if not errors:
                rmse = 0.0
            elif len(errors) == 1:
                rmse = abs(errors[0])
            else:
                rmse = math.sqrt(
                    sum(error * error for error in errors) / len(errors)
                )
        safety_stock = sigma_multiplier * rmse * math.sqrt(
            lead_time if lead_time > 0 else 1
        )
        safety_values.append(safety_stock)
    safety_per_period = [value / window for value in safety_values]
    return pd.DataFrame(
        {
            "ds": decisions_df["ds"].values,
            "safety_stock": safety_values,
            "safety_stock_per_period": safety_per_period,
            "aggregation_window": window,
        }
    )


def _safety_stock_series(
    rows_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
    *,
    unique_id: str | None,
    aggregate: bool,
) -> pd.DataFrame | None:
    if "sigma" not in decisions_df.columns:
        return None
    if aggregate:
        pieces = []
        for article_id in decisions_df["unique_id"].dropna().unique():
            row_subset = rows_df.loc[rows_df["unique_id"] == article_id]
            decision_subset = decisions_df.loc[
                decisions_df["unique_id"] == article_id
            ]
            series = _safety_stock_for_decisions(
                row_subset,
                decision_subset,
                unique_id=article_id,
            )
            if series is not None:
                pieces.append(series)
        if not pieces:
            return None
        combined = pd.concat(pieces, ignore_index=True)
        combined["ds"] = pd.to_datetime(combined["ds"])
        return (
            combined.groupby("ds", as_index=False)[
                ["safety_stock", "safety_stock_per_period"]
            ]
            .sum()
        )
    if unique_id is None:
        return None
    row_subset = rows_df.loc[rows_df["unique_id"] == unique_id]
    decision_subset = decisions_df.loc[decisions_df["unique_id"] == unique_id]
    return _safety_stock_for_decisions(
        row_subset,
        decision_subset,
        unique_id=unique_id,
    )


def _aggregation_window_for_decisions(
    decisions: pd.DataFrame,
    *,
    unique_id: str | None,
    aggregate: bool,
) -> tuple[int | None, dict[str, int] | None]:
    if "aggregation_window" not in decisions.columns:
        return None, None
    window_values = pd.to_numeric(
        decisions["aggregation_window"], errors="coerce"
    ).dropna()
    if window_values.empty:
        return None, None
    if not aggregate and unique_id is not None:
        window_values = window_values.loc[
            decisions["unique_id"] == unique_id
        ]
        window_values = pd.to_numeric(window_values, errors="coerce").dropna()
    if window_values.empty:
        return None, None
    unique_windows = sorted({int(value) for value in window_values})
    if len(unique_windows) > 1:
        if aggregate:
            window_map: dict[str, int] = {}
            grouped = decisions.groupby("unique_id")
            for article_id, group in grouped:
                values = pd.to_numeric(
                    group["aggregation_window"], errors="coerce"
                ).dropna()
                if values.empty:
                    continue
                unique_values = sorted({int(value) for value in values})
                if len(unique_values) > 1:
                    warnings.warn(
                        "Multiple aggregation windows found for unique_id "
                        f"'{article_id}'; using the first value."
                    )
                window_map[article_id] = unique_values[0]
            if window_map:
                return None, window_map
        warnings.warn(
            "Multiple aggregation windows found in decisions; "
            "plots will use raw rows without window aggregation."
        )
        return None, None
    window = unique_windows[0]
    return (window if window > 1 else None), None


def _aggregate_rows_by_window(
    rows: pd.DataFrame,
    *,
    window: int,
    unique_id: str | None,
) -> pd.DataFrame:
    rows = rows.copy()
    rows["ds"] = pd.to_datetime(rows["ds"])
    grouped = (
        rows.groupby("unique_id")
        if unique_id is None
        else [(unique_id, rows.loc[rows["unique_id"] == unique_id])]
    )
    aggregated_rows: list[pd.DataFrame] = []
    for uid, group in grouped:
        if group.empty:
            continue
        group = group.sort_values("ds").reset_index(drop=True)
        group["period_index"] = range(len(group))
        group["agg_index"] = group["period_index"] // window
        ds_map = group.groupby("agg_index")["ds"].min()
        actuals = pd.to_numeric(group["actuals"], errors="coerce")
        forecast = pd.to_numeric(group["forecast"], errors="coerce")
        agg_actuals = actuals.groupby(group["agg_index"]).sum(min_count=1)
        agg_forecast = forecast.groupby(group["agg_index"]).sum(min_count=1)
        aggregated_rows.append(
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": ds_map.values,
                    "actuals": agg_actuals.values,
                    "forecast": agg_forecast.values,
                }
            )
        )
    if not aggregated_rows:
        return rows
    return pd.concat(aggregated_rows, ignore_index=True)


def _aggregate_rows_by_window_map(
    rows: pd.DataFrame,
    *,
    window_map: dict[str, int],
) -> pd.DataFrame:
    aggregated_rows: list[pd.DataFrame] = []
    for article_id, window in window_map.items():
        if window <= 1:
            subset = rows.loc[rows["unique_id"] == article_id]
            if subset.empty:
                continue
            aggregated_rows.append(subset.loc[:, ["unique_id", "ds", "actuals", "forecast"]])
            continue
        aggregated_rows.append(
            _aggregate_rows_by_window(
                rows,
                window=window,
                unique_id=article_id,
            )
        )
    if not aggregated_rows:
        return rows
    return pd.concat(aggregated_rows, ignore_index=True)


def plot_replenishment_decisions(
    rows: Iterable[StandardSimulationRow] | pd.DataFrame,
    decisions: Iterable[ReplenishmentDecisionRow] | pd.DataFrame,
    *,
    unique_id: str | None = None,
    aggregate: bool = False,
    ax: plt.Axes | None = None,
    title: str | None = None,
    decision_style: Literal["bar", "line"] = "bar",
) -> plt.Axes:
    """Plot actuals, forecast, and replenishment decisions for a SKU or aggregate."""
    rows_df = _normalize_rows(rows)
    decisions_df = _normalize_decisions(decisions)
    rows_df = rows_df.copy()
    decisions_df = decisions_df.copy()
    rows_df["ds"] = pd.to_datetime(rows_df["ds"])
    decisions_df["ds"] = pd.to_datetime(decisions_df["ds"])
    aggregated_decisions = False
    if not decisions_df.empty:
        if aggregate:
            for article_id, group in decisions_df.groupby("unique_id"):
                if group.empty:
                    continue
                min_ds = group["ds"].min()
                max_ds = group["ds"].max()
                row_count = len(
                    rows_df.loc[
                        (rows_df["unique_id"] == article_id)
                        & (rows_df["ds"] >= min_ds)
                        & (rows_df["ds"] <= max_ds)
                    ]
                )
                if row_count and len(group) < row_count:
                    aggregated_decisions = True
                    break
        elif unique_id is not None:
            subset = decisions_df.loc[decisions_df["unique_id"] == unique_id]
            if not subset.empty:
                min_ds = subset["ds"].min()
                max_ds = subset["ds"].max()
                row_count = len(
                    rows_df.loc[
                        (rows_df["unique_id"] == unique_id)
                        & (rows_df["ds"] >= min_ds)
                        & (rows_df["ds"] <= max_ds)
                    ]
                )
                decision_count = len(subset)
                aggregated_decisions = bool(
                    row_count and decision_count < row_count
                )

    window = None
    window_map = None
    if aggregated_decisions:
        window, window_map = _aggregation_window_for_decisions(
            decisions_df, unique_id=unique_id, aggregate=aggregate
        )
    timeseries_rows = rows_df

    series = _prepare_timeseries(
        timeseries_rows, decisions_df, unique_id=unique_id, aggregate=aggregate
    )
    decisions_df = decisions_df.copy()
    decisions_df["ds"] = pd.to_datetime(decisions_df["ds"])
    if aggregate:
        start_date = decisions_df["ds"].min() if not decisions_df.empty else None
    else:
        subset = decisions_df.loc[decisions_df["unique_id"] == unique_id, "ds"]
        start_date = subset.min() if not subset.empty else None
    rows_df = rows_df.copy()
    rows_df["ds"] = pd.to_datetime(rows_df["ds"])
    initial_stock = _initial_stock_level(
        rows_df,
        unique_id=unique_id,
        aggregate=aggregate,
        start_date=start_date,
    )
    stock_series, lost_sales = _build_stock_series(
        series, initial_stock, start_date=start_date
    )
    uses_current_stock = False
    uses_ending_stock = False
    if "ending_stock" in decisions_df.columns:
        ending_values = pd.to_numeric(
            decisions_df["ending_stock"], errors="coerce"
        )
        ending_df = decisions_df.assign(ending_stock=ending_values)
        if aggregate:
            ending_df = (
                ending_df.groupby("ds", as_index=False)["ending_stock"]
                .sum()
            )
        else:
            ending_df = ending_df.loc[
                ending_df["unique_id"] == unique_id, ["ds", "ending_stock"]
            ]
        ending_df["ds"] = pd.to_datetime(ending_df["ds"])
        ending_df = ending_df.sort_values("ds")
        stock_series = (
            ending_df.set_index("ds")["ending_stock"]
            .reindex(series["ds"])
            .ffill()
        )
        uses_ending_stock = True
    elif "current_stock" in decisions_df.columns:
        current_values = pd.to_numeric(
            decisions_df["current_stock"], errors="coerce"
        )
        current_df = decisions_df.assign(current_stock=current_values)
        if aggregate:
            current_df = (
                current_df.groupby("ds", as_index=False)["current_stock"]
                .sum()
            )
        else:
            current_df = current_df.loc[
                current_df["unique_id"] == unique_id, ["ds", "current_stock"]
            ]
        current_df["ds"] = pd.to_datetime(current_df["ds"])
        current_df = current_df.sort_values("ds")
        stock_series = (
            current_df.set_index("ds")["current_stock"]
            .reindex(series["ds"])
            .ffill()
        )
        uses_current_stock = True
    elif "ending_on_hand" in decisions_df.columns and "incoming_stock" not in series.columns:
        end_values = pd.to_numeric(
            decisions_df["ending_on_hand"], errors="coerce"
        )
        end_df = decisions_df.assign(ending_on_hand=end_values)
        if aggregate:
            end_df = (
                end_df.groupby("ds", as_index=False)["ending_on_hand"]
                .sum()
            )
        else:
            end_df = end_df.loc[
                end_df["unique_id"] == unique_id, ["ds", "ending_on_hand"]
            ]
        end_df["ds"] = pd.to_datetime(end_df["ds"])
        end_df = end_df.sort_values("ds")
        stock_series = (
            end_df.set_index("ds")["ending_on_hand"]
            .reindex(series["ds"])
            .ffill()
        )

    loss_series: pd.Series
    loss_plot: pd.DataFrame | None = None
    if "missed_sales" in decisions_df.columns:
        missed_sales = pd.to_numeric(
            decisions_df["missed_sales"], errors="coerce"
        ).fillna(0)
        if aggregate:
            loss_plot = (
                decisions_df.assign(missed_sales=missed_sales)
                .groupby("ds", as_index=False)["missed_sales"]
                .sum()
            )
        else:
            loss_plot = decisions_df.loc[
                decisions_df["unique_id"] == unique_id, ["ds"]
            ].copy()
            loss_plot["missed_sales"] = missed_sales.loc[
                decisions_df["unique_id"] == unique_id
            ]
            print(loss_plot["missed_sales"])
        loss_plot["ds"] = pd.to_datetime(loss_plot["ds"])
        loss_plot = loss_plot.sort_values("ds")
        loss_series = loss_plot.set_index("ds")["missed_sales"].reindex(
            series["ds"], fill_value=0
        )
    else:
        loss_series = lost_sales.fillna(0)
    loss_visible = bool(loss_series.any())
    show_loss = aggregate or loss_visible

    start_stock_series: pd.Series | None = None
    start_stock_column: str | None = None
    start_stock_label = "Starting stock"
    start_plot_df: pd.DataFrame | None = None
    show_start_stock = False
    if "starting_stock" in decisions_df.columns:
        start_values = pd.to_numeric(
            decisions_df["starting_stock"], errors="coerce"
        )
        start_df = decisions_df.assign(starting_stock=start_values)
        if aggregate:
            start_df = (
                start_df.groupby("ds", as_index=False)["starting_stock"]
                .sum()
            )
        else:
            start_df = start_df.loc[
                start_df["unique_id"] == unique_id, ["ds", "starting_stock"]
            ]
        start_df["ds"] = pd.to_datetime(start_df["ds"])
        start_df = start_df.sort_values("ds")
        start_plot_df = start_df
        start_stock_column = "starting_stock"
        start_stock_series = start_df.set_index("ds")["starting_stock"].reindex(
            series["ds"]
        )
        show_start_stock = bool(start_stock_series.notna().any())
    elif "starting_on_hand" in decisions_df.columns:
        start_values = pd.to_numeric(
            decisions_df["starting_on_hand"], errors="coerce"
        )
        start_df = decisions_df.assign(starting_on_hand=start_values)
        if aggregate:
            start_df = (
                start_df.groupby("ds", as_index=False)["starting_on_hand"]
                .sum()
            )
        else:
            start_df = start_df.loc[
                start_df["unique_id"] == unique_id, ["ds", "starting_on_hand"]
            ]
        start_df["ds"] = pd.to_datetime(start_df["ds"])
        start_df = start_df.sort_values("ds")
        start_plot_df = start_df
        start_stock_column = "starting_on_hand"
        start_stock_series = start_df.set_index("ds")["starting_on_hand"].reindex(
            series["ds"]
        )
        show_start_stock = bool(start_stock_series.notna().any())
        start_stock_label = "Starting on hand"

    ax_start = None
    ax_main = ax
    ax_loss = None
    ax_cum = None
    ax_share = None
    if ax_main is None:
        if show_start_stock and show_loss:
            _, (ax_start, ax_main, ax_loss, ax_cum, ax_share) = plt.subplots(
                nrows=5,
                sharex=True,
                figsize=(10, 13),
                gridspec_kw={"height_ratios": [2, 3, 1, 1, 1]},
            )
        elif show_start_stock:
            _, (ax_start, ax_main) = plt.subplots(
                nrows=2,
                sharex=True,
                figsize=(10, 7),
                gridspec_kw={"height_ratios": [2, 3]},
            )
        elif show_loss:
            _, (ax_main, ax_loss, ax_cum, ax_share) = plt.subplots(
                nrows=4,
                sharex=True,
                figsize=(10, 11),
                gridspec_kw={"height_ratios": [3, 1, 1, 1]},
            )
        else:
            _, ax_main = plt.subplots(figsize=(10, 5))

    if ax_start is not None and start_plot_df is not None and start_stock_column:
        ax_start.plot(series["ds"], series["actuals"], label="Actuals", marker="o")
        ax_start.plot(series["ds"], series["forecast"], label="Forecast", linestyle="--")
        ax_start.plot(
            start_plot_df["ds"],
            start_plot_df[start_stock_column],
            label=start_stock_label,
            linestyle="-.",
            marker="o",
        )
        ax_start.set_ylabel("Units")
        ax_start.legend()

    ax_main.plot(series["ds"], series["actuals"], label="Actuals", marker="o")
    ax_main.plot(series["ds"], series["forecast"], label="Forecast", linestyle="--")
    forecast_target_df: pd.DataFrame | None = None
    if uses_ending_stock:
        stock_label = "Ending stock"
    elif uses_current_stock:
        stock_label = "Current stock"
    else:
        stock_label = "Projected stock"
    ax_main.plot(series["ds"], stock_series, label=stock_label, linestyle="-.")
    if "forecast_quantity" in decisions_df.columns:
        forecast_values = pd.to_numeric(
            decisions_df["forecast_quantity"], errors="coerce"
        )
        forecast_lead_time_values = None
        if "forecast_quantity_lead_time" in decisions_df.columns:
            forecast_lead_time_values = pd.to_numeric(
                decisions_df["forecast_quantity_lead_time"], errors="coerce"
            )
        window_values = None
        has_agg_window = False
        if "aggregation_window" in decisions_df.columns:
            window_values = pd.to_numeric(
                decisions_df["aggregation_window"], errors="coerce"
            ).fillna(1)
            has_agg_window = bool((window_values > 1).any())
        else:
            window_values = pd.Series(1, index=decisions_df.index)
        if forecast_lead_time_values is not None and not decisions_df.empty:
            forecast_lead_time_values = forecast_lead_time_values
        forecast_df = decisions_df.assign(
            forecast_quantity=forecast_values,
            aggregation_window=window_values,
        )
        if forecast_lead_time_values is not None:
            forecast_df = forecast_df.assign(
                forecast_quantity_lead_time=forecast_lead_time_values
            )
        forecast_df["forecast_per_period"] = forecast_df["forecast_quantity"]
        if aggregate:
            agg_columns = ["forecast_quantity", "forecast_per_period"]
            if "forecast_quantity_lead_time" in forecast_df.columns:
                agg_columns.append("forecast_quantity_lead_time")
            forecast_df = (
                forecast_df.groupby("ds", as_index=False)[agg_columns].sum()
            )
        else:
            forecast_df = forecast_df.loc[
                forecast_df["unique_id"] == unique_id,
                [
                    "ds",
                    "forecast_quantity",
                    "forecast_per_period",
                    "aggregation_window",
                ]
                + (
                    ["forecast_quantity_lead_time"]
                    if "forecast_quantity_lead_time" in forecast_df.columns
                    else []
                ),
            ]
        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
        forecast_df = forecast_df.sort_values("ds")
        if forecast_df["forecast_quantity"].notna().any():
            forecast_target_df = forecast_df
    safety_stock_df = _safety_stock_series(
        rows_df,
        decisions_df,
        unique_id=unique_id,
        aggregate=aggregate,
    )
    if forecast_target_df is not None and safety_stock_df is not None:
        safety_stock_df = safety_stock_df.sort_values("ds")
        safety_plot = forecast_target_df.merge(
            safety_stock_df, on="ds", how="left"
        )
        safety_plot["safety_stock"] = safety_plot["safety_stock"].fillna(0)
        safety_plot["safety_stock_per_period"] = safety_plot[
            "safety_stock_per_period"
        ].fillna(0)
        if "forecast_quantity_lead_time" in safety_plot.columns:
            safety_base = safety_plot["forecast_quantity_lead_time"]
            safety_label = "Forecast + safety stock (lead time)"
        else:
            safety_base = safety_plot["forecast_quantity"]
            safety_label = "Forecast + safety stock"
        ax_main.plot(
            safety_plot["ds"],
            safety_base + safety_plot["safety_stock_per_period"],
            label=safety_label,
            linestyle="dashdot",
            marker="s",
        )
    replenishment = series["replenishment"].fillna(0)
    decision_rows = series.loc[replenishment != 0]
    if decision_style == "line":
        ax_main.vlines(
            decision_rows["ds"],
            0,
            decision_rows["replenishment"],
            label="Replenishment decision",
            alpha=0.4,
        )
        ax_main.scatter(
            decision_rows["ds"],
            decision_rows["replenishment"],
            s=18,
            color=ax_main.lines[-1].get_color(),
            zorder=3,
        )
    else:
        median_step = (
            series["ds"].sort_values().diff().dt.total_seconds().median()
        )
        if pd.isna(median_step) or median_step <= 0:
            bar_width = 1.0
        else:
            bar_width = (median_step / 86400.0) * 0.6
        ax_main.bar(
            decision_rows["ds"],
            decision_rows["replenishment"],
            label="Replenishment decision",
            width=bar_width,
            alpha=0.3,
        )

    ax_main.set_xlabel("Date")
    ax_main.set_ylabel("Units")
    if title is None:
        title = "Aggregate Replenishment Decisions" if aggregate else f"Replenishment for {unique_id}"
    ax_main.set_title(title)
    ax_main.legend()
    if show_loss and ax_loss is not None:
        plot_data = loss_plot
        if plot_data is not None:
            plot_data = plot_data.loc[plot_data["missed_sales"] > 0]
        if plot_data is None or plot_data.empty:
            plot_data = pd.DataFrame(
                {"ds": series["ds"], "missed_sales": loss_series}
            )
        median_step = (
            series["ds"].sort_values().diff().dt.total_seconds().median()
        )
        if pd.isna(median_step) or median_step <= 0:
            bar_width = 1.0
        else:
            bar_width = (median_step / 86400.0) * 0.6
        ax_loss.bar(
            plot_data["ds"],
            plot_data["missed_sales"],
            color="tab:red",
            alpha=0.35,
            width=bar_width,
            label="Lost sales",
        )
        ax_loss.set_ylabel("Lost sales")
        ax_loss.legend()
    if show_loss and ax_cum is not None:
        cumulative_loss = loss_series.cumsum()
        ax_cum.plot(
            series["ds"],
            cumulative_loss,
            color="tab:red",
            label="Cumulative lost sales",
        )
        ax_cum.set_ylabel("Cumulative")
        ax_cum.legend()
    if show_loss and ax_share is not None:
        demand_series = series["forecast"].where(
            series["forecast"].notna(), series["actuals"]
        )
        demand_series = demand_series.fillna(0).astype(float)
        demand_series = pd.Series(demand_series.values, index=series["ds"])
        cumulative_demand = demand_series.cumsum()
        cumulative_loss = loss_series.cumsum()
        loss_share = cumulative_loss.divide(
            cumulative_demand.where(cumulative_demand != 0, pd.NA)
        ).fillna(0) * 100
        ax_share.plot(
            series["ds"],
            loss_share,
            color="tab:red",
            label="Cumulative lost sales share",
            marker="o",
        )
        ax_share.set_ylabel("Share (%)")
        max_share = float(loss_share.max()) if not loss_share.empty else 0.0
        ax_share.set_ylim(0, max(1.0, max_share * 1.1))
        ax_share.legend()
    return ax_main
