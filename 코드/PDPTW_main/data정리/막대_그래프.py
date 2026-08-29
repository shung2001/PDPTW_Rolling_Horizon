"""Create a per-vehicle time bar chart for every available scenario.

The default scan covers Revenue folders from 50,000 through 200,000 in
10,000-unit increments.  A missing Revenue folder is skipped.  Within each
available Revenue folder, every directory named ``vehicles_<number>`` is
processed and two charts are saved:

* ``vehicles_<number>/막대그래프.png``: three separate panels
* ``vehicles_<number>/다중막대그래프.png``: three side-by-side bars per vehicle

The chart uses these fields:

* ``total_ground_time_min``
* ``passenger_flight_time_min``
* ``empty_flight_time_min``

Some existing ``vehicle_summary.csv`` files do not yet contain the two flight
time fields.  In that case they are reconstructed from ``vehicle_route_legs.csv``
using ``travel_time`` and ``movement_type``.  The reconstructed sum is checked
against ``total_flight_time_min`` when that field is available.
"""

from __future__ import annotations

import argparse
import math
import re
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
    REPOSITORY_ROOT
    / "자료"
    / "결과"
    / "Ortools"
    / "d2000"
    / "Penalty_Per_Vehicles"
)

DEFAULT_REVENUE_START = 50_000
DEFAULT_REVENUE_STOP = 200_000
DEFAULT_REVENUE_STEP = 10_000

SUMMARY_NAME = "vehicle_summary.csv"
LEGS_NAME = "vehicle_route_legs.csv"
OUTPUT_NAME = "막대그래프.png"
GROUPED_OUTPUT_NAME = "다중막대그래프.png"
VEHICLE_DIR_PATTERN = re.compile(r"^vehicles_(\d+)$")

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
        description=(
            "Create vehicle-level ground/passenger-flight/empty-flight time "
            "bar charts for all available Revenue and vehicle-count folders."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Penalty_Per_Vehicles directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--revenue-start",
        type=int,
        default=DEFAULT_REVENUE_START,
        help=f"First Revenue value, inclusive (default: {DEFAULT_REVENUE_START})",
    )
    parser.add_argument(
        "--revenue-stop",
        type=int,
        default=DEFAULT_REVENUE_STOP,
        help=f"Last Revenue value, inclusive (default: {DEFAULT_REVENUE_STOP})",
    )
    parser.add_argument(
        "--revenue-step",
        type=int,
        default=DEFAULT_REVENUE_STEP,
        help=f"Revenue range step (default: {DEFAULT_REVENUE_STEP})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output PNG resolution (default: 180)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.revenue_step <= 0:
        raise ValueError(f"revenue-step must be positive: {args.revenue_step}")
    if args.revenue_start > args.revenue_stop:
        raise ValueError(
            "revenue-start must be less than or equal to revenue-stop: "
            f"{args.revenue_start} > {args.revenue_stop}"
        )
    if args.dpi <= 0:
        raise ValueError(f"dpi must be positive: {args.dpi}")


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


def vehicle_count_from_dir(vehicle_dir: Path) -> int:
    match = VEHICLE_DIR_PATTERN.fullmatch(vehicle_dir.name)
    if match is None:
        raise ValueError(f"Invalid vehicle directory name: {vehicle_dir}")
    return int(match.group(1))


def vehicle_directories(revenue_dir: Path) -> list[Path]:
    directories = [
        path
        for path in revenue_dir.iterdir()
        if path.is_dir() and VEHICLE_DIR_PATTERN.fullmatch(path.name)
    ]
    return sorted(directories, key=vehicle_count_from_dir)


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
    revenue: int,
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
        f"Vehicle Time Breakdown | Revenue {revenue:,} | {vehicle_count} Vehicles",
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
    revenue: int,
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
        f"Vehicle Time Comparison | Revenue {revenue:,} | "
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


def revenue_values(start: int, stop: int, step: int) -> range:
    """Return an inclusive Revenue range."""
    return range(start, stop + 1, step)


def main() -> None:
    args = parse_args()
    validate_args(args)
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    created: list[Path] = []
    skipped_revenues: list[int] = []
    empty_revenues: list[int] = []

    for revenue in revenue_values(
        args.revenue_start, args.revenue_stop, args.revenue_step
    ):
        revenue_dir = data_dir / str(revenue)
        if not revenue_dir.is_dir():
            skipped_revenues.append(revenue)
            continue

        directories = vehicle_directories(revenue_dir)
        if not directories:
            empty_revenues.append(revenue)
            continue

        for vehicle_dir in directories:
            frame = load_vehicle_times(vehicle_dir)
            vehicle_count = vehicle_count_from_dir(vehicle_dir)

            output_path = vehicle_dir / OUTPUT_NAME
            save_bar_chart(
                frame=frame,
                revenue=revenue,
                vehicle_count=vehicle_count,
                output_path=output_path,
                dpi=args.dpi,
            )
            created.append(output_path)
            print(f"created: {output_path}")

            grouped_output_path = vehicle_dir / GROUPED_OUTPUT_NAME
            save_grouped_bar_chart(
                frame=frame,
                revenue=revenue,
                vehicle_count=vehicle_count,
                output_path=grouped_output_path,
                dpi=args.dpi,
            )
            created.append(grouped_output_path)
            print(f"created: {grouped_output_path}")

    print(f"charts_created={len(created)}")
    print(f"missing_revenue_folders_skipped={len(skipped_revenues)}")
    if skipped_revenues:
        print("missing_revenues=" + ",".join(map(str, skipped_revenues)))
    if empty_revenues:
        print("revenues_without_vehicle_folders=" + ",".join(map(str, empty_revenues)))
    if not created:
        raise RuntimeError("No charts were created for the requested Revenue range")


if __name__ == "__main__":
    main()
