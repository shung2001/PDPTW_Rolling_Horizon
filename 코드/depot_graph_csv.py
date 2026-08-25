# -*- coding: utf-8 -*-
"""Depot별 CSV에서 시간대별 차량 대수 heatmap을 생성한다.

각 CSV에는 ``time_min``과 ``vehicle_count`` 열이 필수다.
CSV에 없는 분(minute)은 해당 Depot의 차량이 0대인 것으로 처리한다.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = (
    BASE_DIR
    / "자료"
    / "결과"
    / "Ortools"
    / "8월21일"
    / "FOCUSED"
    / "Remove_Depot"
    / "100000"
    / "분류된_노드"
)

# 출력할 Depot 순서와 실제 CSV 파일명
DEPOT_FILES = {
    "서울역": "서울역.csv",
    "수서": "수서.csv",
    "삼성": "삼성.csv",
    "여의도": "여의도.csv",
    "김포공항": "김포공항.csv",
    "일산": "일산.csv",
    "판교": "판교.csv",
    "동탄/용인": "동탄_용인.csv",
    "평택": "평택.csv",
    "인천공항": "인천공항.csv",
}

DEFAULT_BASE_TIME = "05:40"
DEFAULT_TICK_INTERVAL_MIN = 120
DEFAULT_OUTPUT_NAME = "depot_vehicle_count_spectrum.png"
REQUIRED_COLUMNS = {"time_min", "vehicle_count"}


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Depot별 CSV를 읽어 시간대별 차량 대수 heatmap을 만듭니다."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Depot CSV 폴더 (기본값: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "저장할 PNG 경로 "
            f"(기본값: <data-dir>/{DEFAULT_OUTPUT_NAME})"
        ),
    )
    parser.add_argument(
        "--base-time",
        default=DEFAULT_BASE_TIME,
        help=f"time_min=0인 시각, HH:MM 형식 (기본값: {DEFAULT_BASE_TIME})",
    )
    parser.add_argument(
        "--tick-interval",
        type=int,
        default=DEFAULT_TICK_INTERVAL_MIN,
        help=(
            "x축 시간 라벨 간격(분) "
            f"(기본값: {DEFAULT_TICK_INTERVAL_MIN})"
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="그래프 창을 열지 않고 PNG만 저장합니다.",
    )
    return parser.parse_args(argv)


def read_depot_csv(file_path: Path) -> pd.Series:
    """CSV 한 개를 읽어 time_min 인덱스의 vehicle_count를 반환한다."""
    if not file_path.is_file():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {file_path}")

    # utf-8-sig는 BOM이 있는 UTF-8과 없는 UTF-8 모두 읽을 수 있다.
    try:
        frame = pd.read_csv(file_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        # 기존 Windows/Excel에서 만든 CSV를 위한 호환 처리
        frame = pd.read_csv(file_path, encoding="cp949")

    # 열 이름 앞뒤의 실수로 들어간 공백을 허용한다.
    frame.columns = frame.columns.astype(str).str.strip()
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise KeyError(
            f"{file_path.name}에 필수 열이 없습니다: "
            f"{sorted(missing_columns)}; 실제 열: {list(frame.columns)}"
        )

    values = frame[["time_min", "vehicle_count"]].copy()
    values["time_min"] = pd.to_numeric(values["time_min"], errors="coerce")
    values["vehicle_count"] = pd.to_numeric(
        values["vehicle_count"], errors="coerce"
    )
    values = values.dropna(subset=["time_min", "vehicle_count"])

    if values.empty:
        raise ValueError(
            f"{file_path.name}에 유효한 time_min/vehicle_count 데이터가 없습니다."
        )

    fractional_times = values["time_min"] % 1 != 0
    if fractional_times.any():
        bad_values = values.loc[fractional_times, "time_min"].head(5).tolist()
        raise ValueError(
            f"{file_path.name}의 time_min은 정수여야 합니다: {bad_values}"
        )

    if (values["time_min"] < 0).any():
        raise ValueError(f"{file_path.name}의 time_min에 음수가 있습니다.")
    if (values["vehicle_count"] < 0).any():
        raise ValueError(f"{file_path.name}의 vehicle_count에 음수가 있습니다.")

    values["time_min"] = values["time_min"].astype(int)
    # 동일한 time_min이 여러 행이면 차량 대수를 합산한다.
    return values.groupby("time_min", sort=True)["vehicle_count"].sum()


def load_depot_data(data_dir: Path) -> dict[str, pd.Series]:
    if not data_dir.is_dir():
        raise NotADirectoryError(f"CSV 폴더를 찾을 수 없습니다: {data_dir}")

    missing_files = [
        file_name
        for file_name in DEPOT_FILES.values()
        if not (data_dir / file_name).is_file()
    ]
    if missing_files:
        formatted = "\n  - ".join(missing_files)
        raise FileNotFoundError(
            f"다음 Depot CSV 파일이 없습니다:\n  - {formatted}\n"
            f"폴더: {data_dir}"
        )

    return {
        depot_name: read_depot_csv(data_dir / file_name)
        for depot_name, file_name in DEPOT_FILES.items()
    }


def make_matrix(
    depot_series: dict[str, pd.Series],
) -> tuple[np.ndarray, np.ndarray]:
    all_times = [
        int(time_min)
        for series in depot_series.values()
        for time_min in series.index
    ]
    if not all_times:
        raise ValueError("유효한 time_min 데이터가 없습니다.")

    time_axis = np.arange(min(all_times), max(all_times) + 1, dtype=int)
    matrix = np.vstack(
        [
            series.reindex(time_axis, fill_value=0).to_numpy(dtype=float)
            for series in depot_series.values()
        ]
    )
    return matrix, time_axis


def parse_base_time(value: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%H:%M")
    except ValueError as exc:
        raise ValueError(
            f"base-time은 HH:MM 형식이어야 합니다: {value!r}"
        ) from exc
    return parsed.replace(year=2000, month=1, day=1)


def plot_heatmap(
    matrix: np.ndarray,
    time_axis: np.ndarray,
    depot_names: Sequence[str],
    base_datetime: datetime,
    tick_interval_min: int,
) -> tuple[plt.Figure, plt.Axes]:
    if tick_interval_min <= 0:
        raise ValueError("tick-interval은 1 이상의 정수여야 합니다.")

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    min_time = int(time_axis[0])
    max_time = int(time_axis[-1])
    tick_times = np.arange(min_time, max_time + 1, tick_interval_min)
    tick_positions = tick_times - min_time
    tick_labels = [
        (base_datetime + timedelta(minutes=int(value))).strftime("%H:%M")
        for value in tick_times
    ]

    figure, axis = plt.subplots(figsize=(16, 7))
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )
    axis.set_yticks(np.arange(len(depot_names)))
    axis.set_yticklabels(depot_names)
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels)
    axis.set_xlabel("시간")
    axis.set_ylabel("Depot")
    axis.set_title("시간대별 Depot 차량 대수 스펙트럼")

    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("차량 대수")
    figure.tight_layout()
    return figure, axis


def print_peak_counts(
    matrix: np.ndarray,
    time_axis: np.ndarray,
    depot_names: Sequence[str],
    base_datetime: datetime,
) -> None:
    print("\n[Depot별 최대 vehicle_count]")
    for row_index, depot_name in enumerate(depot_names):
        row = matrix[row_index]
        max_index = int(row.argmax())
        max_count = row[max_index]
        peak_time_min = int(time_axis[max_index])
        peak_time_hhmm = (
            base_datetime + timedelta(minutes=peak_time_min)
        ).strftime("%H:%M")

        # 입력이 정수일 때는 불필요한 .0을 표시하지 않는다.
        count_text = (
            str(int(max_count)) if float(max_count).is_integer() else f"{max_count:g}"
        )
        print(f"{depot_name:8s} : 최대 {count_text:>2s}대 ({peak_time_hhmm})")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    data_dir = args.data_dir.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else data_dir / DEFAULT_OUTPUT_NAME
    )
    base_datetime = parse_base_time(args.base_time)

    depot_series = load_depot_data(data_dir)
    matrix, time_axis = make_matrix(depot_series)
    depot_names = list(depot_series)
    figure, _ = plot_heatmap(
        matrix,
        time_axis,
        depot_names,
        base_datetime,
        args.tick_interval,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"그래프 저장 완료: {output_path}")
    print_peak_counts(matrix, time_axis, depot_names, base_datetime)

    if args.no_show:
        plt.close(figure)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
