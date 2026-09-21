"""Revenue별 차량 수에 따른 Penalty/Overdue 선 그래프를 생성한다.

기본 입력 폴더 구조는 다음과 같다.

```
자료/결과/Ortools/VP_기존/Rolling_Horizon_구간_30/d3000/single/
└── <Revenue>/Original/add_6mins/<차량 수>/
    └── rolling_horizon_request_status.csv
```

기본적으로 Revenue 110,000부터 130,000까지 10,000 간격으로 확인한다.
존재하지 않는 Revenue 폴더나 ``Original/add_6mins`` 결과 경로는 건너뛴다.
각 차량 수 폴더의 최종 요청 상태에서 ``Overdue``인 행만 골라 다음 값을
집계한다.

* ``total_penalty``: Overdue 행의 패널티 합계
* ``overdue_batch_count``: Overdue 행 개수

모든 Revenue를 각각 하나의 선으로 표시한 다음 두 PNG는 ``--data-dir``에
저장한다.

* ``penalty_by_num_vehicles.png``
* ``overdue_by_num_vehicles.png``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# GUI가 없는 환경에서도 실행할 수 있도록 파일 출력용 백엔드를 사용한다.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

CONFIG = {'DEMAND': 5000}

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = (
    REPOSITORY_ROOT
    / "자료"
    / "결과"
    / "Ortools"
    / "VP_기존"
    / "Rolling_Horizon_구간_30"
    / f"d{CONFIG['DEMAND']}"
    / "single"
)
DEFAULT_RESULT_PATH = Path("Original") / "add_6mins"

DEFAULT_REVENUE_START = 110_000
DEFAULT_REVENUE_STOP = 130_000
DEFAULT_REVENUE_STEP = 10_000

STATUS_NAME = "rolling_horizon_request_status.csv"
STATUS_COLUMN = "final_status"
PENALTY_COLUMN_CANDIDATES = ("final_drop_penalty", "final_cost(원)")
OVERDUE_STATUS = "Overdue"

VEHICLE_COLUMN = "num_vehicles"
PENALTY_COLUMN = "total_penalty"
OVERDUE_COLUMN = "overdue_batch_count"
PENALTY_OUTPUT_NAME = f"penalty_by_num_vehicles_{CONFIG['DEMAND']}.png"
OVERDUE_OUTPUT_NAME = f"overdue_by_num_vehicles_{CONFIG['DEMAND']}.png"


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
            "각 Revenue의 차량 수별 최종 요청 상태를 읽어 Penalty와 "
            "Overdue 선 그래프를 각각 PNG로 생성합니다."
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
        default=200,
        help="출력 PNG 해상도 (기본값: 200)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.result_path.is_absolute():
        raise ValueError(
            f"result-path는 Revenue 폴더 기준 상대 경로여야 합니다: "
            f"{args.result_path}"
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


def revenue_values(start: int, stop: int, step: int) -> range:
    """마지막 값을 포함하는 Revenue 범위를 반환한다."""
    return range(start, stop + 1, step)


def vehicle_count_from_dir(vehicle_dir: Path) -> int:
    if not vehicle_dir.name.isdecimal():
        raise ValueError(f"차량 수 폴더 이름이 숫자가 아닙니다: {vehicle_dir}")
    return int(vehicle_dir.name)


def vehicle_directories(result_dir: Path) -> list[Path]:
    """숫자 이름의 차량 수 폴더를 차량 수 오름차순으로 반환한다."""
    directories = [
        path
        for path in result_dir.iterdir()
        if path.is_dir() and path.name.isdecimal()
    ]
    return sorted(directories, key=vehicle_count_from_dir)


def penalty_column_name(status: pd.DataFrame, status_path: Path) -> str:
    """사용할 수 있는 최종 패널티 열 이름을 반환한다."""
    for column in PENALTY_COLUMN_CANDIDATES:
        if column in status.columns:
            return column
    candidates = ", ".join(PENALTY_COLUMN_CANDIDATES)
    raise ValueError(f"{status_path}: 패널티 열이 없습니다: {candidates}")


def load_vehicle_result(vehicle_dir: Path) -> dict[str, int | float]:
    """차량 수 폴더 하나에서 Overdue 건수와 패널티 합계를 집계한다."""
    status_path = vehicle_dir / STATUS_NAME
    if not status_path.is_file():
        raise FileNotFoundError(f"최종 요청 상태 파일이 없습니다: {status_path}")

    status = pd.read_csv(status_path, encoding="utf-8-sig")
    if status.empty:
        raise ValueError(f"{status_path}: 요청 상태 데이터가 없습니다")
    if STATUS_COLUMN not in status.columns:
        raise ValueError(f"{status_path}: 필수 열이 없습니다: {STATUS_COLUMN}")
    if status[STATUS_COLUMN].isna().any():
        rows = (status[STATUS_COLUMN].isna().loc[lambda values: values].index + 2).tolist()
        raise ValueError(
            f"{status_path}: {STATUS_COLUMN} 값이 없는 CSV 행: {rows}"
        )

    penalty_column = penalty_column_name(status, status_path)
    overdue = status[STATUS_COLUMN].eq(OVERDUE_STATUS)
    try:
        penalties = pd.to_numeric(
            status.loc[overdue, penalty_column], errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{status_path}: {penalty_column} 열에 숫자가 아닌 값이 있습니다"
        ) from error

    invalid = penalties.isna() | ~np.isfinite(penalties)
    if invalid.any():
        rows = (invalid.loc[lambda values: values].index + 2).tolist()
        raise ValueError(
            f"{status_path}: {penalty_column} 값이 잘못된 CSV 행: {rows}"
        )
    if (penalties < 0).any():
        rows = ((penalties < 0).loc[lambda values: values].index + 2).tolist()
        raise ValueError(
            f"{status_path}: {penalty_column} 값이 음수인 CSV 행: {rows}"
        )

    return {
        VEHICLE_COLUMN: vehicle_count_from_dir(vehicle_dir),
        PENALTY_COLUMN: penalties.sum(),
        OVERDUE_COLUMN: int(overdue.sum()),
    }


def load_revenue_results(result_dir: Path) -> pd.DataFrame:
    """Revenue 하나의 모든 차량 수 결과를 읽는다."""
    directories = vehicle_directories(result_dir)
    if not directories:
        raise FileNotFoundError(
            f"{result_dir}: 숫자 이름의 차량 수 폴더가 없습니다"
        )
    return pd.DataFrame(load_vehicle_result(path) for path in directories)


def discover_revenue_data(
    data_dir: Path,
    result_path: Path,
    revenue_start: int,
    revenue_stop: int,
    revenue_step: int,
) -> tuple[dict[int, pd.DataFrame], list[int], list[Path], list[Path]]:
    """요청한 Revenue 범위의 결과와 건너뛴 경로 정보를 반환한다."""
    revenue_data: dict[int, pd.DataFrame] = {}
    missing_revenues: list[int] = []
    missing_result_paths: list[Path] = []
    empty_result_paths: list[Path] = []

    for revenue in revenue_values(revenue_start, revenue_stop, revenue_step):
        revenue_dir = data_dir / str(revenue)
        if not revenue_dir.is_dir():
            missing_revenues.append(revenue)
            continue

        result_dir = revenue_dir / result_path
        if not result_dir.is_dir():
            missing_result_paths.append(result_dir)
            continue

        if not vehicle_directories(result_dir):
            empty_result_paths.append(result_dir)
            continue

        revenue_data[revenue] = load_revenue_results(result_dir)

    return (
        revenue_data,
        missing_revenues,
        missing_result_paths,
        empty_result_paths,
    )


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
            markersize=5,
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


def generate_charts(
    data_dir: Path,
    result_path: Path = DEFAULT_RESULT_PATH,
    revenue_start: int = DEFAULT_REVENUE_START,
    revenue_stop: int = DEFAULT_REVENUE_STOP,
    revenue_step: int = DEFAULT_REVENUE_STEP,
    dpi: int = 200,
) -> list[Path]:
    """두 그래프를 생성하고 생성된 파일 경로를 반환한다."""
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"기준 데이터 폴더가 없습니다: {data_dir}")
    if result_path.is_absolute():
        raise ValueError(
            f"result-path는 Revenue 폴더 기준 상대 경로여야 합니다: "
            f"{result_path}"
        )
    if revenue_step <= 0:
        raise ValueError(f"revenue-step은 양수여야 합니다: {revenue_step}")
    if revenue_start > revenue_stop:
        raise ValueError(
            "revenue-start는 revenue-stop보다 작거나 같아야 합니다: "
            f"{revenue_start} > {revenue_stop}"
        )
    if dpi <= 0:
        raise ValueError(f"dpi는 양수여야 합니다: {dpi}")

    (
        revenue_data,
        missing_revenues,
        missing_result_paths,
        empty_result_paths,
    ) = discover_revenue_data(
        data_dir,
        result_path,
        revenue_start,
        revenue_stop,
        revenue_step,
    )
    if not revenue_data:
        raise RuntimeError("요청한 Revenue 범위에서 그래프로 만들 결과가 없습니다")

    created: list[Path] = []
    for spec in CHART_SPECS:
        output_path = data_dir / spec.output_name
        save_line_chart(revenue_data, spec, output_path, dpi)
        created.append(output_path)
        print(f"created: {output_path}")

    print(f"revenues_plotted={len(revenue_data)}")
    print("revenue_values=" + ",".join(map(str, revenue_data)))
    print(f"missing_revenue_folders_skipped={len(missing_revenues)}")
    if missing_revenues:
        print("missing_revenues=" + ",".join(map(str, missing_revenues)))
    if missing_result_paths:
        print("missing_result_paths:")
        for path in missing_result_paths:
            print(f"  {path}")
    if empty_result_paths:
        print("result_paths_without_numeric_vehicle_folders:")
        for path in empty_result_paths:
            print(f"  {path}")
    print(f"charts_created={len(created)}")
    return created


def main() -> None:
    args = parse_args()
    validate_args(args)
    generate_charts(
        data_dir=args.data_dir,
        result_path=args.result_path,
        revenue_start=args.revenue_start,
        revenue_stop=args.revenue_stop,
        revenue_step=args.revenue_step,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
