"""모든 OD 요청의 발생 빈도를 하나의 시간 스펙트럼으로 시각화한다.

상태와 OD 조합을 구분하지 않고 전체 요청을 합산하여 time_window_start_hhmm
기준 시간 구간별 발생 건수를 하나의 가로 띠에 표시한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


PDPTW_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    PDPTW_ROOT
    / "자료"
    / "결과"
    / "Ortools"
    / "d5000"
    / "Penalty_Per_Vehicles"
    / "Original"
    / "130000"
    / "vehicles_150"
    / "rolling_horizon_request_status.csv"
)
REQUIRED_COLUMNS = {"origin", "destination", "time_window_start_hhmm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="모든 OD 요청을 합산한 시간대별 발생 스펙트럼을 생성합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="출력 PNG 경로 (기본값: 입력 CSV 폴더의 *_all_od_spectrum.png)",
    )
    parser.add_argument(
        "--time-bin", type=int, default=10, help="집계 간격(분, 기본값: 10)"
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def build_spectrum(data: pd.DataFrame, time_bin: int) -> tuple[np.ndarray, np.ndarray]:
    start_times = data["time_window_start_hhmm"]
    if start_times.empty:
        raise ValueError("입력 파일에 요청 행이 없습니다.")

    parsed = pd.to_datetime(
        start_times.astype("string").str.strip(), format="%H:%M", errors="coerce"
    )
    if parsed.isna().any():
        rows = start_times.index[parsed.isna()].tolist()[:10]
        raise ValueError(f"시작 시각이 잘못된 요청 행이 있습니다: {rows}")

    start_minutes = parsed.dt.hour * 60 + parsed.dt.minute
    grouped_minutes = (start_minutes // time_bin) * time_bin
    first_bin = int(grouped_minutes.min())
    last_bin = int(grouped_minutes.max())
    time_bins = np.arange(first_bin, last_bin + time_bin, time_bin)
    counts = grouped_minutes.value_counts().reindex(time_bins, fill_value=0)
    return time_bins, counts.to_numpy(dtype=int)


def save_spectrum(
    time_bins: np.ndarray,
    counts: np.ndarray,
    time_bin: int,
    output_path: Path,
    dpi: int,
) -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    maximum = max(1, int(counts.max()))
    reds = plt.colormaps["YlOrRd"].resampled(maximum + 1)
    colors = [(1, 1, 1, 1)] + [reds(i) for i in range(1, maximum + 1)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, maximum + 1.5), cmap.N)

    fig, ax = plt.subplots(figsize=(17, 4.2))
    image = ax.imshow(
        counts[np.newaxis, :],
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=[time_bins[0], time_bins[-1] + time_bin, 0, 1],
    )

    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlim(time_bins[0], time_bins[-1] + time_bin)
    ax.set_xlabel(f"Time window 시작 시각 ({time_bin}분 간격)")
    ax.set_title(
        "전체 OD 요청 통합 Time Window 시작 스펙트럼\n"
        f"총 {int(counts.sum()):,}건 · 진한 색일수록 발생 빈도가 높음",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    tick_interval = max(30, time_bin)
    first_tick = int(np.ceil(time_bins[0] / tick_interval) * tick_interval)
    ticks = np.arange(first_tick, time_bins[-1] + time_bin, tick_interval)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{minute // 60:02d}:{minute % 60:02d}" for minute in ticks],
        rotation=45,
        ha="right",
    )
    ax.grid(axis="x", color="black", alpha=0.15, linewidth=0.6)

    # 빈도가 있는 구간에는 발생 건수를 직접 표시한다.
    for minute, count in zip(time_bins, counts):
        if count == 0:
            continue
        color = "white" if count >= max(2, maximum * 0.6) else "black"
        ax.text(
            minute + time_bin / 2,
            0.5,
            str(count),
            ha="center",
            va="center",
            fontsize=7,
            color=color,
        )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        pad=0.42,
        fraction=0.18,
        ticks=np.arange(0, maximum + 1, 10),
    )
    colorbar.set_label("시간 구간별 전체 요청 발생 빈도")
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
        else input_path.with_name(f"{input_path.stem}_all_od_spectrum.png")
    )
    if output_path.suffix.casefold() != ".png":
        raise ValueError("--output 경로의 확장자는 .png여야 합니다.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path, dtype=str, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise KeyError(f"입력 CSV에 필수 컬럼이 없습니다: {sorted(missing)}")

    time_bins, counts = build_spectrum(data, args.time_bin)
    save_spectrum(time_bins, counts, args.time_bin, output_path, args.dpi)
    print(f"전체 OD 요청 건수: {int(counts.sum()):,}건")
    print(f"PNG 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
