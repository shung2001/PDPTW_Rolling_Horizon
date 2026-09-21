"""Overdue 요청의 시작 시각과 방향별 OD를 PNG 히트맵으로 표시한다.

각 요청은 ``origin`` → ``destination``의 한 행에 집계한다. 시간대는
``time_window_start_hhmm``을 기준으로 하며, 실제 Overdue 판정 시각이 아니다.
셀의 색과 숫자는 해당 시간대에 시작한 Overdue 요청 건수이다.
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
    / "VP_기존"
    / "Rolling_Horizon_구간_30"
    / "d5000"
    / "single"
    / "130000"
    / "Original"
    / "add_6mins"
    / "150"
    / "rolling_horizon_request_status.csv"
)
REQUIRED_COLUMNS = {"final_status", "origin", "destination", "time_window_start_hhmm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overdue 요청의 시작 시간대와 방향별 OD 빈도를 히트맵으로 저장합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        help="출력 PNG 경로 (기본값: 입력 CSV 폴더의 *_overdue_od.png)",
    )
    parser.add_argument("--time-bin", type=int, default=10, help="시간대 간격(분, 기본값: 10)")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def node_sort_key(value: str) -> tuple[int, float | str]:
    """숫자 노드는 숫자 순, 나머지는 문자 순으로 정렬한다."""
    try:
        return 0, float(value)
    except ValueError:
        return 1, value


def prepare_frequency_table(data: pd.DataFrame, time_bin: int) -> pd.DataFrame:
    """행은 방향별 OD, 열은 시간대 시작 시각(분), 값은 Overdue 건수."""
    status = data["final_status"].fillna("").str.strip().str.casefold()
    overdue = data.loc[
        status.eq("overdue"), ["origin", "destination", "time_window_start_hhmm"]
    ].copy()
    if overdue.empty:
        raise ValueError("입력 CSV에 final_status가 Overdue인 행이 없습니다.")

    for column in ("origin", "destination"):
        overdue[column] = overdue[column].astype("string").str.strip()
    parsed = pd.to_datetime(
        overdue["time_window_start_hhmm"].astype("string").str.strip(),
        format="%H:%M",
        errors="coerce",
    )
    invalid = (
        overdue["origin"].isna()
        | overdue["origin"].eq("")
        | overdue["destination"].isna()
        | overdue["destination"].eq("")
        | parsed.isna()
    )
    if invalid.any():
        raise ValueError(f"OD 또는 시작 시각이 잘못된 Overdue 행: {overdue.index[invalid].tolist()[:10]}")

    start_minutes = parsed.dt.hour * 60 + parsed.dt.minute
    overdue["time_bin_minutes"] = (start_minutes // time_bin) * time_bin
    pairs = sorted(
        set(zip(overdue["origin"], overdue["destination"])),
        key=lambda pair: (node_sort_key(pair[0]), node_sort_key(pair[1])),
    )
    pair_index = pd.MultiIndex.from_tuples(pairs, names=["origin", "destination"])
    first_bin = int(overdue["time_bin_minutes"].min())
    last_bin = int(overdue["time_bin_minutes"].max())
    bins = list(range(first_bin, last_bin + time_bin, time_bin))
    counts = overdue.groupby(["origin", "destination", "time_bin_minutes"]).size()
    return counts.unstack("time_bin_minutes", fill_value=0).reindex(
        index=pair_index, columns=bins, fill_value=0
    )


def save_heatmap(frequency: pd.DataFrame, output: Path, time_bin: int, dpi: int) -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    values = frequency.to_numpy(dtype=int)
    maximum = max(1, int(values.max()))
    cmap = plt.colormaps["YlOrRd"].resampled(maximum + 1)
    cmap.set_under("white")
    norm = BoundaryNorm(np.arange(0.5, maximum + 1.5), cmap.N)

    fig_width = max(14.0, 0.24 * frequency.shape[1])
    fig_height = max(6.0, 0.45 * frequency.shape[0] + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    label_step = max(1, int(np.ceil(30 / time_bin)))
    positions = np.arange(0, frequency.shape[1], label_step)
    minutes = frequency.columns.to_numpy()[positions]
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{minute // 60:02d}:{minute % 60:02d}" for minute in minutes],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(frequency.shape[0]))
    ax.set_yticklabels(
        [f"{origin} → {destination}" for origin, destination in frequency.index],
        fontsize=7.5,
    )
    ax.set_xlabel(f"Overdue 요청 시작 시간대 ({time_bin}분 간격)")
    ax.set_ylabel("Origin → Destination")
    ax.set_title(
        f"시간대별·OD별 Overdue 요청 빈도 (총 {int(values.sum()):,}건)",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    for row, column in np.argwhere(values > 0):
        count = int(values[row, column])
        color = "white" if count >= max(2, maximum * 0.6) else "black"
        ax.text(column, row, str(count), ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(np.arange(-0.5, frequency.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, frequency.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("Overdue 요청 건수")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
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
        else input_path.with_name(f"{input_path.stem}_overdue_od.png")
    )
    if output_path.suffix.casefold() != ".png":
        raise ValueError("--output 경로의 확장자는 .png여야 합니다.")

    data = pd.read_csv(input_path, dtype=str, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise KeyError(f"입력 CSV에 필수 컬럼이 없습니다: {sorted(missing)}")
    frequency = prepare_frequency_table(data, args.time_bin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_heatmap(frequency, output_path, args.time_bin, args.dpi)
    print(f"Overdue 총 건수: {int(frequency.to_numpy().sum()):,}건")
    print(f"OD 조합: {frequency.shape[0]:,}개")
    print(f"PNG 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
