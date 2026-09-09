"""Create two vehicle time charts from one result folder or vehicle_summary.csv."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

# The script is intended for batch use, including headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import StrMethodFormatter


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = (
    REPOSITORY_ROOT/
    "자료" / 
    "결과" / 
    "Ortools" / 
    "Rolling_Horizon_구간_30" / 
    "d5000" / 
    "Penalty_Per_Vehicles" / 
    "Time_Solver_120" / 
    "add_6mins_penalty_수정_2" /
    "Original")


SUMMARY_NAME = "vehicle_summary.csv"
LEGS_NAME = "vehicle_route_legs.csv"
OUTPUT_NAME = "막대그래프.png"
GROUPED_OUTPUT_NAME = "다중막대그래프.png"

GROUND_COLUMN = "total_ground_time_min"
PASSENGER_COLUMN = "passenger_flight_time_min"
EMPTY_COLUMN = "empty_flight_time_min"
PLOT_COLUMNS = (GROUND_COLUMN, PASSENGER_COLUMN, EMPTY_COLUMN)

SERIES_LABELS = {
    GROUND_COLUMN: "Ground time",
    PASSENGER_COLUMN: "Passenger flight time",
    EMPTY_COLUMN: "Empty flight time",
}
SERIES_COLORS = {
    GROUND_COLUMN: "#4C78A8",
    PASSENGER_COLUMN: "#F58518",
    EMPTY_COLUMN: "#54A24B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create two vehicle time charts from one result folder."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Result folder or vehicle_summary.csv path",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args

def numeric_series(frame: pd.DataFrame, column: str, csv_path: Path) -> pd.Series:
    """Return a finite, non-negative numeric column or raise a clear error."""
    if column not in frame.columns:
        raise ValueError(f"{csv_path}: missing required column: {column}")

    values = pd.to_numeric(frame[column], errors="raise")
    invalid = values.isna() | ~np.isfinite(values)
    if invalid.any():
        rows = (invalid[invalid].index + 2).tolist()
        raise ValueError(f"{csv_path}: {column} has invalid values at CSV rows {rows}")
    if (values < 0).any():
        rows = ((values < 0)[values < 0].index + 2).tolist()
        raise ValueError(f"{csv_path}: {column} has negative values at CSV rows {rows}")
    return values


def flight_times_from_legs(legs_path: Path) -> pd.DataFrame:
    """Aggregate passenger and empty flight minutes by vehicle ID."""
    if not legs_path.is_file():
        raise FileNotFoundError(
            f"Missing {LEGS_NAME}; it is required because the summary has no "
            f"{PASSENGER_COLUMN}/{EMPTY_COLUMN}: {legs_path}"
        )

    legs = pd.read_csv(legs_path)
    required = {"vehicle_id", "event", "movement_type", "travel_time"}
    missing = required.difference(legs.columns)
    if missing:
        raise ValueError(
            f"{legs_path}: missing required columns: {', '.join(sorted(missing))}"
        )

    legs["vehicle_id"] = pd.to_numeric(legs["vehicle_id"], errors="raise")
    legs["travel_time"] = pd.to_numeric(legs["travel_time"], errors="raise")
    if legs[["vehicle_id", "travel_time"]].isna().any().any():
        raise ValueError(f"{legs_path}: vehicle_id/travel_time contains missing values")
    if (~np.isfinite(legs["travel_time"]) | (legs["travel_time"] < 0)).any():
        raise ValueError(f"{legs_path}: travel_time must be finite and non-negative")

    movement = legs.loc[legs["event"].eq("Movement")].copy()
    known_types = {"passenger_flight", "empty_repositioning", "return_to_depot"}
    unknown_types = sorted(set(movement["movement_type"].dropna()) - known_types)
    if unknown_types:
        raise ValueError(
            f"{legs_path}: unclassified Movement types: {', '.join(unknown_types)}"
        )

    passenger = (
        movement.loc[movement["movement_type"].eq("passenger_flight")]
        .groupby("vehicle_id")["travel_time"]
        .sum()
    )
    empty = (
        movement.loc[
            movement["movement_type"].isin(["empty_repositioning", "return_to_depot"])
        ]
        .groupby("vehicle_id")["travel_time"]
        .sum()
    )

    vehicle_ids = pd.Index(movement["vehicle_id"].unique(), name="vehicle_id")
    result = pd.DataFrame(index=vehicle_ids)
    result[PASSENGER_COLUMN] = passenger.reindex(vehicle_ids, fill_value=0.0)
    result[EMPTY_COLUMN] = empty.reindex(vehicle_ids, fill_value=0.0)
    return result


def load_vehicle_times(vehicle_dir: Path) -> pd.DataFrame:
    """Load the three chart series, deriving flight splits when necessary."""
    summary_path = vehicle_dir / SUMMARY_NAME
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing input file: {summary_path}")

    frame = pd.read_csv(summary_path)
    if frame.empty:
        raise ValueError(f"{summary_path}: no vehicle rows")

    frame["vehicle_id"] = numeric_series(frame, "vehicle_id", summary_path)
    frame[GROUND_COLUMN] = numeric_series(frame, GROUND_COLUMN, summary_path)
    if frame["vehicle_id"].duplicated().any():
        duplicates = frame.loc[frame["vehicle_id"].duplicated(False), "vehicle_id"].tolist()
        raise ValueError(f"{summary_path}: duplicate vehicle_id values: {duplicates}")

    has_flight_columns = all(column in frame.columns for column in (PASSENGER_COLUMN, EMPTY_COLUMN))
    if has_flight_columns:
        frame[PASSENGER_COLUMN] = numeric_series(frame, PASSENGER_COLUMN, summary_path)
        frame[EMPTY_COLUMN] = numeric_series(frame, EMPTY_COLUMN, summary_path)
    else:
        flight_times = flight_times_from_legs(vehicle_dir / LEGS_NAME)
        frame = frame.join(flight_times, on="vehicle_id")
        frame[[PASSENGER_COLUMN, EMPTY_COLUMN]] = frame[
            [PASSENGER_COLUMN, EMPTY_COLUMN]
        ].fillna(0.0)

    if "total_flight_time_min" in frame.columns:
        total_flight = numeric_series(frame, "total_flight_time_min", summary_path)
        split_total = frame[PASSENGER_COLUMN] + frame[EMPTY_COLUMN]
        mismatch = ~np.isclose(total_flight, split_total, rtol=0.0, atol=1e-9)
        if mismatch.any():
            vehicle_ids = frame.loc[mismatch, "vehicle_id"].tolist()
            raise ValueError(
                f"{vehicle_dir}: passenger + empty flight time does not equal "
                f"total_flight_time_min for vehicle_id values {vehicle_ids}"
            )

    return frame.loc[:, ["vehicle_id", *PLOT_COLUMNS]].sort_values("vehicle_id")


def label_step(vehicle_count: int) -> int:
    """Keep roughly 25 x-axis labels even for the largest scenarios."""
    return max(1, math.ceil(vehicle_count / 25))


def configure_axis(ax: Axes, title: str) -> None:
    ax.set_title(title, fontsize=12, loc="left", pad=6)
    ax.set_ylabel("Minutes", fontsize=10)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)


def save_bar_chart(
    frame: pd.DataFrame,
    vehicle_count: int,
    output_path: Path,
    dpi: int,
) -> None:
    """Save three aligned bar panels so small flight values remain readable."""
    row_count = len(frame)
    width = min(30.0, max(14.0, row_count * 0.19))
    fig, axes = plt.subplots(3, 1, figsize=(width, 12), sharex=True)
    x = np.arange(row_count)

    for ax, column in zip(axes, PLOT_COLUMNS, strict=True):
        ax.bar(
            x,
            frame[column].to_numpy(),
            width=0.78,
            color=SERIES_COLORS[column],
            edgecolor="white",
            linewidth=0.25,
        )
        configure_axis(ax, SERIES_LABELS[column])

    step = label_step(row_count)
    tick_positions = x[::step]
    tick_labels = frame["vehicle_id"].iloc[::step].map(lambda value: f"{value:g}")
    axes[-1].set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
    axes[-1].set_xlabel("Vehicle ID", fontsize=10)
    for ax in axes:
        ax.set_xlim(-0.7, row_count - 0.3)

    fig.suptitle(
        f"Vehicle Time Breakdown | {vehicle_count} Vehicles",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_grouped_bar_chart(
    frame: pd.DataFrame,
    vehicle_count: int,
    output_path: Path,
    dpi: int,
) -> None:
    """Save all three time series together as grouped bars by vehicle."""
    row_count = len(frame)
    width = min(30.0, max(14.0, row_count * 0.19))
    fig, ax = plt.subplots(figsize=(width, 8))
    x = np.arange(row_count)
    bar_width = 0.25

    for index, column in enumerate(PLOT_COLUMNS):
        offset = (index - (len(PLOT_COLUMNS) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            frame[column].to_numpy(),
            width=bar_width,
            label=SERIES_LABELS[column],
            color=SERIES_COLORS[column],
            edgecolor="white",
            linewidth=0.25,
        )

    step = label_step(row_count)
    tick_positions = x[::step]
    tick_labels = frame["vehicle_id"].iloc[::step].map(lambda value: f"{value:g}")
    ax.set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
    ax.set_xlim(-0.7, row_count - 0.3)
    ax.set_xlabel("Vehicle ID", fontsize=10)
    configure_axis(ax, "")
    ax.legend(title="Time type")
    ax.set_title(
        f"Vehicle Time Comparison | "
        f"{vehicle_count} Vehicles",
        fontsize=16,
        fontweight="bold",
        loc="center",
        pad=12,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    vehicle_dir = args.data_dir.resolve()
    if vehicle_dir.is_file():
        if vehicle_dir.name != SUMMARY_NAME:
            raise ValueError(f"Expected {SUMMARY_NAME}: {vehicle_dir}")
        vehicle_dir = vehicle_dir.parent
    if not vehicle_dir.is_dir():
        raise FileNotFoundError(f"Result directory not found: {vehicle_dir}")

    frame = load_vehicle_times(vehicle_dir)
    for save_chart, name in (
        (save_bar_chart, OUTPUT_NAME),
        (save_grouped_bar_chart, GROUPED_OUTPUT_NAME),
    ):
        output_path = vehicle_dir / name
        save_chart(
            frame=frame, vehicle_count=len(frame),
            output_path=output_path, dpi=args.dpi,
        )
        print(f"created: {output_path}")
    print(f"vehicles={len(frame)}, charts_created=2")


if __name__ == "__main__":
    main()