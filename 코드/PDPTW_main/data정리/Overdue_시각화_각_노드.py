"""시작 시각별·출발 노드별 Overdue 발생 빈도를 PNG 히트맵으로 저장한다.

Overdue가 발생한 노드는 요청의 ``origin``으로 정의하고, 발생 시각은
``time_window_start_hhmm``을 사용한다. 각 셀의 색과 숫자는 해당 시간 구간에
해당 노드에서 시작한 Overdue 요청의 수를 나타낸다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


PDPTW_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    PDPTW_ROOT
    / "자료"
    / "결과"
    / "Ortools"
    / "Rolling_Horizon_구간_60"
    / "d5000"
    / "Penalty_Per_Vehicles"
    / "Time_Solver_60"
    / "130000"
    / "vehicles_150"
    / "rolling_horizon_request_status.csv"
)
REQUIRED_COLUMNS = {"final_status", "origin", "time_window_start_hhmm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="시작 시각별·출발 노드별 Overdue 빈도 히트맵을 생성합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="출력 PNG 경로 (기본값: 입력 CSV 폴더의 *_overdue_by_node.png)",
    )
    parser.add_argument(
        "--time-bin",
        type=int,
        default=10,
        help="시작 시각 집계 간격(분, 기본값: 10)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def natural_node_key(value: object) -> tuple[int, float | str]:
    """숫자 노드는 숫자 순으로, 그 밖의 노드는 문자 순으로 정렬한다."""
    text = str(value).strip()
    try:
        return 0, float(text)
    except ValueError:
        return 1, text


def prepare_frequency_table(data: pd.DataFrame, time_bin: int) -> pd.DataFrame:
    status = data["final_status"].fillna("").str.strip().str.casefold()
    overdue = data.loc[status.eq("overdue"), ["origin", "time_window_start_hhmm"]].copy()
    if overdue.empty:
        raise ValueError("입력 파일에 final_status가 Overdue인 행이 없습니다.")

    overdue["origin"] = overdue["origin"].astype("string").str.strip()
    parsed = pd.to_datetime(
        overdue["time_window_start_hhmm"].astype("string").str.strip(),
        format="%H:%M",
        errors="coerce",
    )
    invalid = parsed.isna() | overdue["origin"].isna() | overdue["origin"].eq("")
    if invalid.any():
        bad_rows = overdue.index[invalid].tolist()[:10]
        raise ValueError(f"노드 또는 시작 시각이 잘못된 Overdue 행이 있습니다: {bad_rows}")

    start_minutes = parsed.dt.hour * 60 + parsed.dt.minute
    overdue["time_bin_minutes"] = (start_minutes // time_bin) * time_bin

    nodes = sorted(overdue["origin"].unique(), key=natural_node_key)
    first_bin = int(overdue["time_bin_minutes"].min())
    last_bin = int(overdue["time_bin_minutes"].max())
    all_bins = list(range(first_bin, last_bin + time_bin, time_bin))

    frequency = pd.crosstab(overdue["origin"], overdue["time_bin_minutes"])
    return frequency.reindex(index=nodes, columns=all_bins, fill_value=0)


def save_heatmap(
    frequency: pd.DataFrame,
    output_path: Path,
    time_bin: int,
    dpi: int,
) -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    values = frequency.to_numpy(dtype=int)
    maximum = max(1, int(values.max()))
    cmap = plt.colormaps["YlOrRd"].resampled(maximum + 1)
    cmap.set_under("white")
    norm = BoundaryNorm(np.arange(0.5, maximum + 1.5), cmap.N)

    figure_width = max(14.0, 0.24 * frequency.shape[1])
    figure_height = max(6.0, 0.55 * frequency.shape[0] + 2.5)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    # 시간 라벨은 과밀하지 않도록 약 30분 이상의 간격으로 표시한다.
    label_step = max(1, int(np.ceil(30 / time_bin)))
    tick_positions = np.arange(0, frequency.shape[1], label_step)
    tick_minutes = frequency.columns.to_numpy()[tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [f"{minute // 60:02d}:{minute % 60:02d}" for minute in tick_minutes],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(frequency.shape[0]))
    ax.set_yticklabels([f"Node {node}" for node in frequency.index])
    ax.set_xlabel(f"Overdue 시작 시각 ({time_bin}분 간격)")
    ax.set_ylabel("출발 노드 (origin)")
    ax.set_title(
        "Overdue 발생 시각 및 노드별 빈도\n"
        f"총 {int(values.sum()):,}건",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    # 0이 아닌 셀에 실제 빈도를 직접 표시한다.
    for row, column in np.argwhere(values > 0):
        count = values[row, column]
        text_color = "white" if count >= max(2, maximum * 0.6) else "black"
        ax.text(column, row, str(count), ha="center", va="center", fontsize=7, color=text_color)

    ax.set_xticks(np.arange(-0.5, frequency.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, frequency.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.015,
        ticks=np.arange(1, maximum + 1),
    )
    colorbar.set_label("Overdue 발생 빈도")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.time_bin <= 0 or 60 % args.time_bin != 0:
        raise ValueError("--time-bin은 60의 약수인 양의 정수여야 합니다.")
    if args.dpi <= 0:
        raise ValueError("--dpi는 양의 정수여야 합니다.")

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"입력 CSV를 찾을 수 없습니다: {input_path}")
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_overdue_by_node.png")
    )
    if output_path.suffix.casefold() != ".png":
        raise ValueError("--output 경로의 확장자는 .png여야 합니다.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path, dtype=str, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise KeyError(f"입력 CSV에 필수 컬럼이 없습니다: {sorted(missing)}")

    frequency = prepare_frequency_table(data, args.time_bin)
    save_heatmap(frequency, output_path, args.time_bin, args.dpi)
    print(f"Overdue 총 발생 건수: {int(frequency.to_numpy().sum()):,}건")
    print(f"분석 노드 수: {frequency.shape[0]:,}개")
    print(f"PNG 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
