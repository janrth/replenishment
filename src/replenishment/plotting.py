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

    data = grouped_rows.merge(grouped_decisions, on="ds", how="left")
    data["replenishment"] = data["replenishment"].fillna(0)
    return data.sort_values("ds")


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

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(series["ds"], series["actuals"], label="Actuals", marker="o")
    ax.plot(series["ds"], series["forecast"], label="Forecast", linestyle="--")

    if decision_style == "line":
        ax.plot(
            series["ds"],
            series["replenishment"],
            label="Replenishment",
            linestyle="-.",
        )
    else:
        ax.bar(
            series["ds"],
            series["replenishment"],
            label="Replenishment",
            alpha=0.3,
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    if title is None:
        title = "Aggregate Replenishment Decisions" if aggregate else f"Replenishment for {unique_id}"
    ax.set_title(title)
    ax.legend()
    return ax
