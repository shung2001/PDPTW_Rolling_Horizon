"""차량별 지상·승객 비행·공차 비행 시간 막대그래프를 생성한다.

기본 입력 폴더 구조는 다음과 같다.

```
자료/결과/Ortools/VP_기존/Rolling_Horizon_구간_30/d3000/single/
└── <Revenue>/Original/add_6mins/<차량 수>/
    ├── vehicle_summary.csv
    └── vehicle_route_legs.csv
```

기본적으로 Revenue 110,000부터 130,000까지 10,000 간격으로 확인한다.
존재하지 않는 Revenue 폴더는 건너뛰고, 각 Revenue의
``Original/add_6mins`` 아래에 있는 모든 숫자 이름의 차량 폴더를 처리한다.

각 차량 폴더에는 다음 그래프 두 개가 생성된다.

* ``막대그래프.png``: 시간 종류별로 분리된 세 개의 패널
* ``다중막대그래프.png``: 차량별 세 시간 값을 나란히 표시한 그래프

``vehicle_summary.csv``에 ``passenger_flight_time_min`` 또는
``empty_flight_time_min`` 열이 없으면 ``vehicle_route_legs.csv``의
``travel_time``과 ``movement_type``을 이용해 해당 값을 계산한다.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

# GUI가 없는 환경에서도 실행할 수 있도록 파일 출력용 백엔드를 사용한다.
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
    / "VP_기존"
    / "Rolling_Horizon_구간_30"
    / "d3000"
    / "single"
)
DEFAULT_RESULT_PATH = Path("Original") / "add_6mins"

DEFAULT_REVENUE_START = 110_000
DEFAULT_REVENUE_STOP = 130_000
DEFAULT_REVENUE_STEP = 10_000

SUMMARY_NAME = "vehicle_summary.csv"
LEGS_NAME = "vehicle_route_legs.csv"
OUTPUT_NAME = "막대그래프.png"
GROUPED_OUTPUT_NAME = "다중막대그래프.png"
VEHICLE_DIR_PATTERN = re.compile(r"^(\d+)$")

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
            "bar charts for every available Revenue and vehicle-count folder."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Revenue 폴더들이 있는 기준 경로 (기본값: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=DEFAULT_RESULT_PATH,
        help=(
            "각 Revenue 폴더 아래 결과 상대 경로 "
            f"(기본값: {DEFAULT_RESULT_PATH})"
        ),
    )
    parser.add_argument(
        "--revenue-start",
        type=int,
        default=DEFAULT_REVENUE_START,
        help=f"첫 Revenue 값, 포함 (기본값: {DEFAULT_REVENUE_START})",
    )
    parser.add_argument(
        "--revenue-stop",
        type=int,
        default=DEFAULT_REVENUE_STOP,
        help=f"마지막 Revenue 값, 포함 (기본값: {DEFAULT_REVENUE_STOP})",
    )
    parser.add_argument(
        "--revenue-step",
        type=int,
        default=DEFAULT_REVENUE_STEP,
        help=f"Revenue 간격 (기본값: {DEFAULT_REVENUE_STEP})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="출력 PNG 해상도 (기본값: 180)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.result_path.is_absolute():
        raise ValueError(
            f"result-path는 Revenue 폴더 기준 상대 경로여야 합니다: {args.result_path}"
        )
    if args.revenue_step <= 0:
        raise ValueError(f"revenue-step은 양수여야 합니다: {args.revenue_step}")
    if args.revenue_start > args.revenue_stop:
        raise ValueError(
            "revenue-start는 revenue-stop보다 작거나 같아야 합니다: "
            f"{args.revenue_start} > {args.revenue_stop}"
        )
    if args.dpi <= 0:
        raise ValueError(f"dpi는 양수여야 합니다: {args.dpi}")


def numeric_series(frame: pd.DataFrame, column: str, csv_path: Path) -> pd.Series:
    """유한한 0 이상의 숫자 열을 반환하고, 잘못된 값은 명확히 알린다."""
    if column not in frame.columns:
        raise ValueError(f"{csv_path}: 필수 열이 없습니다: {column}")

    values = pd.to_numeric(frame[column], errors="raise")
    invalid = values.isna() | ~np.isfinite(values)
    if invalid.any():
        rows = (invalid[invalid].index + 2).tolist()
        raise ValueError(f"{csv_path}: {column} 값이 잘못된 CSV 행: {rows}")
    if (values < 0).any():
        rows = ((values < 0)[values < 0].index + 2).tolist()
        raise ValueError(f"{csv_path}: {column} 값이 음수인 CSV 행: {rows}")
    return values


def flight_times_from_legs(legs_path: Path) -> pd.DataFrame:
    """운항 구간 데이터에서 차량별 승객·공차 비행 시간을 집계한다."""
    if not legs_path.is_file():
        raise FileNotFoundError(
            f"{PASSENGER_COLUMN}/{EMPTY_COLUMN} 계산에 필요한 "
            f"{LEGS_NAME} 파일이 없습니다: {legs_path}"
        )

    legs = pd.read_csv(legs_path)
    required = {"vehicle_id", "event", "movement_type", "travel_time"}
    missing = required.difference(legs.columns)
    if missing:
        raise ValueError(
            f"{legs_path}: 필수 열이 없습니다: {', '.join(sorted(missing))}"
        )

    legs["vehicle_id"] = pd.to_numeric(legs["vehicle_id"], errors="raise")
    legs["travel_time"] = pd.to_numeric(legs["travel_time"], errors="raise")
    if legs[["vehicle_id", "travel_time"]].isna().any().any():
        raise ValueError(f"{legs_path}: vehicle_id/travel_time에 결측값이 있습니다")
    if (~np.isfinite(legs["travel_time"]) | (legs["travel_time"] < 0)).any():
        raise ValueError(f"{legs_path}: travel_time은 유한한 0 이상 값이어야 합니다")

    movement = legs.loc[legs["event"].eq("Movement")].copy()
    known_types = {"passenger_flight", "empty_repositioning", "return_to_depot"}
    unknown_types = sorted(set(movement["movement_type"].dropna()) - known_types)
    if unknown_types:
        raise ValueError(
            f"{legs_path}: 분류할 수 없는 Movement 유형: "
            f"{', '.join(unknown_types)}"
        )

    passenger = (
        movement.loc[movement["movement_type"].eq("passenger_flight")]
        .groupby("vehicle_id")["travel_time"]
        .sum()
    )
    empty = (
        movement.loc[
            movement["movement_type"].isin(
                ["empty_repositioning", "return_to_depot"]
            )
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
    """그래프에 사용할 세 시간 값을 읽고, 없는 비행 시간 열은 계산한다."""
    summary_path = vehicle_dir / SUMMARY_NAME
    if not summary_path.is_file():
        raise FileNotFoundError(f"입력 파일이 없습니다: {summary_path}")

    frame = pd.read_csv(summary_path)
    if frame.empty:
        raise ValueError(f"{summary_path}: 차량 데이터가 없습니다")

    frame["vehicle_id"] = numeric_series(frame, "vehicle_id", summary_path)
    frame[GROUND_COLUMN] = numeric_series(frame, GROUND_COLUMN, summary_path)
    if frame["vehicle_id"].duplicated().any():
        duplicates = frame.loc[
            frame["vehicle_id"].duplicated(False), "vehicle_id"
        ].tolist()
        raise ValueError(f"{summary_path}: 중복 vehicle_id 값: {duplicates}")

    missing_flight_columns = [
        column
        for column in (PASSENGER_COLUMN, EMPTY_COLUMN)
        if column not in frame.columns
    ]
    if missing_flight_columns:
        flight_times = flight_times_from_legs(vehicle_dir / LEGS_NAME)
        for column in missing_flight_columns:
            frame[column] = frame["vehicle_id"].map(flight_times[column]).fillna(0.0)

    frame[PASSENGER_COLUMN] = numeric_series(
        frame, PASSENGER_COLUMN, summary_path
    )
    frame[EMPTY_COLUMN] = numeric_series(frame, EMPTY_COLUMN, summary_path)

    if "total_flight_time_min" in frame.columns:
        total_flight = numeric_series(frame, "total_flight_time_min", summary_path)
        split_total = frame[PASSENGER_COLUMN] + frame[EMPTY_COLUMN]
        mismatch = ~np.isclose(total_flight, split_total, rtol=0.0, atol=1e-9)
        if mismatch.any():
            vehicle_ids = frame.loc[mismatch, "vehicle_id"].tolist()
            raise ValueError(
                f"{vehicle_dir}: 승객 비행 시간 + 공차 비행 시간이 "
                "total_flight_time_min과 일치하지 않는 vehicle_id: "
                f"{vehicle_ids}"
            )

    result = frame.loc[:, ["vehicle_id", *PLOT_COLUMNS]].sort_values("vehicle_id")
    return result.reset_index(drop=True)


def vehicle_count_from_dir(vehicle_dir: Path) -> int:
    match = VEHICLE_DIR_PATTERN.fullmatch(vehicle_dir.name)
    if match is None:
        raise ValueError(f"차량 수 폴더 이름이 숫자가 아닙니다: {vehicle_dir}")
    return int(match.group(1))


def vehicle_directories(result_dir: Path) -> list[Path]:
    directories = [
        path
        for path in result_dir.iterdir()
        if path.is_dir() and VEHICLE_DIR_PATTERN.fullmatch(path.name)
    ]
    return sorted(directories, key=vehicle_count_from_dir)


def label_step(vehicle_count: int) -> int:
    """차량 수가 많아도 x축 레이블이 약 25개를 넘지 않게 한다."""
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
    """작은 비행 시간도 보이도록 세 시간 값을 패널별로 저장한다."""
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
    """차량별 세 시간 값을 한 그래프의 다중 막대로 저장한다."""
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
    """마지막 값을 포함하는 Revenue 범위를 반환한다."""
    return range(start, stop + 1, step)


def main() -> None:
    args = parse_args()
    validate_args(args)
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"기준 데이터 폴더가 없습니다: {data_dir}")

    created: list[Path] = []
    skipped_revenues: list[int] = []
    missing_result_paths: list[Path] = []
    empty_result_paths: list[Path] = []

    for revenue in revenue_values(
        args.revenue_start, args.revenue_stop, args.revenue_step
    ):
        revenue_dir = data_dir / str(revenue)
        if not revenue_dir.is_dir():
            skipped_revenues.append(revenue)
            continue

        result_dir = revenue_dir / args.result_path
        if not result_dir.is_dir():
            missing_result_paths.append(result_dir)
            continue

        directories = vehicle_directories(result_dir)
        if not directories:
            empty_result_paths.append(result_dir)
            continue

        for vehicle_dir in directories:
            frame = load_vehicle_times(vehicle_dir)
            vehicle_count = vehicle_count_from_dir(vehicle_dir)
            if len(frame) != vehicle_count:
                raise ValueError(
                    f"{vehicle_dir}: 폴더의 차량 수({vehicle_count})와 "
                    f"vehicle_summary.csv 행 수({len(frame)})가 다릅니다"
                )

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
    if missing_result_paths:
        print("missing_result_paths:")
        for path in missing_result_paths:
            print(f"  {path}")
    if empty_result_paths:
        print("result_paths_without_numeric_vehicle_folders:")
        for path in empty_result_paths:
            print(f"  {path}")
    if not created:
        raise RuntimeError("요청한 Revenue 범위에서 생성된 그래프가 없습니다")


if __name__ == "__main__":
    main()
