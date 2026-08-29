"""Revenue별 차량 수에 따른 Penalty/Overdue 선 그래프를 생성한다.

``Penalty_Per_Vehicles/<Revenue>/penalty_per_vehicles.csv``를 모두 읽고,
각 Revenue를 하나의 선으로 표시한 두 PNG를 ``Penalty_Per_Vehicles``
최상위 폴더에 저장한다.

* ``penalty_by_num_vehicles.png``: (차량 수, total_penalty)
* ``overdue_by_num_vehicles.png``: (차량 수, overdue_batch_count)
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# GUI가 없는 환경에서도 PNG를 생성할 수 있게 한다.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

CSV_NAME = "penalty_per_vehicles.csv"
PENALTY_OUTPUT_NAME = "penalty_by_num_vehicles.png"
OVERDUE_OUTPUT_NAME = "overdue_by_num_vehicles.png"

REVENUE_COLUMN = "revenue"
VEHICLE_COLUMN = "num_vehicles"
PENALTY_COLUMN = "total_penalty"
OVERDUE_COLUMN = "overdue_batch_count"
REQUIRED_COLUMNS = {
    REVENUE_COLUMN,
    VEHICLE_COLUMN,
    PENALTY_COLUMN,
    OVERDUE_COLUMN,
}
REVENUE_DIR_PATTERN = re.compile(r"^\d+$")


@dataclass(frozen=True)
class ChartSpec:
    column: str
    ylabel: str
    title: str
    output_name: str


CHART_SPECS = (
    ChartSpec(
        column=PENALTY_COLUMN,
        ylabel="Penalty",
        title="Penalty by Number of Vehicles",
        output_name=PENALTY_OUTPUT_NAME,
    ),
    ChartSpec(
        column=OVERDUE_COLUMN,
        ylabel="Number of Overdue Batches",
        title="Overdue Batches by Number of Vehicles",
        output_name=OVERDUE_OUTPUT_NAME,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "모든 Revenue의 penalty_per_vehicles.csv를 읽어 차량 수에 따른 "
            "Penalty와 Overdue 선 그래프를 각각 PNG로 생성합니다."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Penalty_Per_Vehicles 폴더 (기본값: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="출력 PNG 해상도 (기본값: 200)",
    )
    return parser.parse_args()


def numeric_column(frame: pd.DataFrame, column: str, csv_path: Path) -> pd.Series:
    """CSV 열을 유한한 숫자로 변환하고 잘못된 행을 명확히 알린다."""
    try:
        values = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{csv_path}: {column} 열에 숫자가 아닌 값이 있습니다.") from error

    invalid = values.isna() | ~np.isfinite(values)
    if invalid.any():
        csv_rows = (invalid[invalid].index + 2).tolist()
        raise ValueError(
            f"{csv_path}: {column} 열의 CSV {csv_rows}행에 유효하지 않은 값이 있습니다."
        )
    return values


def load_revenue_csv(csv_path: Path, directory_revenue: int) -> pd.DataFrame:
    """Revenue CSV 하나를 읽고 그래프에 필요한 열을 검증한다."""
    frame = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{csv_path}: 필수 열이 없습니다: {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise ValueError(f"{csv_path}: 데이터 행이 없습니다.")

    selected = frame.loc[:, sorted(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS:
        selected[column] = numeric_column(selected, column, csv_path)

    non_integer_vehicles = selected[VEHICLE_COLUMN] % 1 != 0
    if non_integer_vehicles.any():
        csv_rows = (non_integer_vehicles[non_integer_vehicles].index + 2).tolist()
        raise ValueError(
            f"{csv_path}: {VEHICLE_COLUMN} 열의 CSV {csv_rows}행이 정수가 아닙니다."
        )
    selected[VEHICLE_COLUMN] = selected[VEHICLE_COLUMN].astype(int)

    if (selected[VEHICLE_COLUMN] < 0).any():
        raise ValueError(f"{csv_path}: {VEHICLE_COLUMN}은 음수일 수 없습니다.")
    if (selected[[PENALTY_COLUMN, OVERDUE_COLUMN]] < 0).any().any():
        raise ValueError(f"{csv_path}: Penalty/Overdue는 음수일 수 없습니다.")

    csv_revenues = selected[REVENUE_COLUMN].unique()
    if len(csv_revenues) != 1 or csv_revenues[0] != directory_revenue:
        raise ValueError(
            f"{csv_path}: 폴더 Revenue({directory_revenue})와 CSV revenue "
            f"값({csv_revenues.tolist()})이 일치하지 않습니다."
        )

    duplicated = selected[VEHICLE_COLUMN].duplicated(keep=False)
    if duplicated.any():
        vehicle_counts = sorted(selected.loc[duplicated, VEHICLE_COLUMN].unique())
        raise ValueError(
            f"{csv_path}: 차량 수가 중복됩니다: {vehicle_counts}"
        )

    return selected.sort_values(VEHICLE_COLUMN).reset_index(drop=True)


def discover_revenue_data(data_dir: Path) -> dict[int, pd.DataFrame]:
    """숫자 이름의 Revenue 하위 폴더를 모두 찾아 CSV를 읽는다."""
    revenue_dirs = sorted(
        (
            path
            for path in data_dir.iterdir()
            if path.is_dir() and REVENUE_DIR_PATTERN.fullmatch(path.name)
        ),
        key=lambda path: int(path.name),
    )
    if not revenue_dirs:
        raise FileNotFoundError(f"{data_dir}: Revenue 하위 폴더가 없습니다.")

    missing_csvs = [path / CSV_NAME for path in revenue_dirs if not (path / CSV_NAME).is_file()]
    if missing_csvs:
        formatted = "\n".join(f"- {path}" for path in missing_csvs)
        raise FileNotFoundError(
            f"다음 Revenue 폴더에 {CSV_NAME}가 없습니다:\n{formatted}"
        )

    return {
        int(revenue_dir.name): load_revenue_csv(
            revenue_dir / CSV_NAME, int(revenue_dir.name)
        )
        for revenue_dir in revenue_dirs
    }


def save_line_chart(
    revenue_data: dict[int, pd.DataFrame],
    spec: ChartSpec,
    output_path: Path,
    dpi: int,
) -> None:
    """Revenue당 하나의 선을 가진 선 그래프를 PNG로 저장한다."""
    fig, ax = plt.subplots(figsize=(12, 7.2))
    colors = plt.get_cmap("tab20", len(revenue_data))

    for color_index, (revenue, frame) in enumerate(revenue_data.items()):
        ax.plot(
            frame[VEHICLE_COLUMN],
            frame[spec.column],
            color=colors(color_index),
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=f"Revenue {revenue:,}",
        )

    all_vehicle_counts = sorted(
        {
            int(vehicle_count)
            for frame in revenue_data.values()
            for vehicle_count in frame[VEHICLE_COLUMN]
        }
    )
    ax.set_xticks(all_vehicle_counts)
    ax.set_xlabel("Number of Vehicles", fontsize=12)
    ax.set_ylabel(spec.ylabel, fontsize=12)
    ax.set_title(spec.title, fontsize=16, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(
        title="Revenue",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=9,
        title_fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_charts(data_dir: Path, dpi: int = 200) -> list[Path]:
    """두 그래프를 생성하고 생성된 파일 경로를 반환한다."""
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"데이터 폴더를 찾을 수 없습니다: {data_dir}")
    if dpi <= 0:
        raise ValueError(f"dpi는 양수여야 합니다: {dpi}")

    revenue_data = discover_revenue_data(data_dir)
    created: list[Path] = []
    for spec in CHART_SPECS:
        output_path = data_dir / spec.output_name
        save_line_chart(revenue_data, spec, output_path, dpi)
        created.append(output_path)
        print(f"created: {output_path}")

    print(f"revenues_plotted={len(revenue_data)}")
    print("revenue_values=" + ",".join(map(str, revenue_data)))
    print(f"charts_created={len(created)}")
    return created


def main() -> None:
    args = parse_args()
    generate_charts(args.data_dir, args.dpi)


if __name__ == "__main__":
    main()
