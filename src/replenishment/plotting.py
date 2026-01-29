"""Plotting helpers for replenishment simulations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

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
        grouped_decisions = (
            decisions.groupby("ds", as_index=False)
            .agg(replenishment=("quantity", "sum"))
            .sort_values("ds")
        )
    else:
        grouped_rows = rows.loc[
            rows["unique_id"] == unique_id, ["ds", "actuals", "forecast"]
        ]
        grouped_decisions = decisions.loc[
            decisions["unique_id"] == unique_id, ["ds", "quantity"]
        ].rename(columns={"quantity": "replenishment"})

    data = grouped_rows.merge(grouped_decisions, on="ds", how="outer")
    data["replenishment"] = data["replenishment"].fillna(0)
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

    series = _prepare_timeseries(
        rows_df, decisions_df, unique_id=unique_id, aggregate=aggregate
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

    loss_series = lost_sales.fillna(0)
    loss_visible = bool(loss_series.any())
    show_loss = aggregate or loss_visible
    ax_loss = None
    if ax is None:
        if show_loss:
            _, (ax, ax_loss) = plt.subplots(
                nrows=2,
                sharex=True,
                figsize=(10, 7),
                gridspec_kw={"height_ratios": [3, 1]},
            )
        else:
            _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(series["ds"], series["actuals"], label="Actuals", marker="o")
    ax.plot(series["ds"], series["forecast"], label="Forecast", linestyle="--")
    ax.plot(series["ds"], stock_series, label="Projected stock", linestyle="-.")
    replenishment = series["replenishment"].fillna(0)
    decision_rows = series.loc[replenishment != 0]
    if decision_style == "line":
        ax.vlines(
            decision_rows["ds"],
            0,
            decision_rows["replenishment"],
            label="Replenishment decision",
            alpha=0.4,
        )
        ax.scatter(
            decision_rows["ds"],
            decision_rows["replenishment"],
            s=18,
            color=ax.lines[-1].get_color(),
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
        ax.bar(
            decision_rows["ds"],
            decision_rows["replenishment"],
            label="Replenishment decision",
            width=bar_width,
            alpha=0.3,
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    if title is None:
        title = "Aggregate Replenishment Decisions" if aggregate else f"Replenishment for {unique_id}"
    ax.set_title(title)
    ax.legend()
    if show_loss and ax_loss is not None:
        ax_loss.bar(
            series["ds"],
            loss_series,
            color="tab:red",
            alpha=0.35,
            label="Lost sales",
        )
        ax_loss.set_ylabel("Lost sales")
        ax_loss.legend()
    return ax
