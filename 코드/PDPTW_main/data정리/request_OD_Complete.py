"""Complete 요청의 time window를 시간대별·OD별 히트맵으로 시각화한다.

final_status가 Complete인 요청만 사용한다(대소문자 및 앞뒤 공백 무시).
실제 운행 시간이 아닌 요청의 time window를 표시한다.
각 요청은 ``time_window_start_hhmm``부터 ``time_window_end_hhmm``까지
실제로 겹치는 모든 시간 구간에 집계된다. 세로축은 origin → destination,
가로축은 시간이며 셀의 색과 숫자는 해당 시간에 활성화된 time window 수이다.
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
    / "Rolling_Horizon_구간_30"
    / "d5000"
    / "Penalty_Per_Vehicles"
    / "Time_Solver_120"
    / "add_6mins_penalty_수정_2"
    / "Original"
    / "rolling_horizon_request_status.csv"
)
REQUIRED_COLUMNS = {
    "final_status",
    "origin",
    "destination",
    "time_window_start_hhmm",
    "time_window_end_hhmm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete 요청의 time window를 시간대별·OD별 PNG 히트맵으로 만듭니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="출력 PNG 경로 (기본값: 입력 CSV 폴더의 *_complete_time_windows_od.png)",
    )
    parser.add_argument(
        "--time-bin",
        type=int,
        default=5,
        help="시간 집계 간격(분, 기본값: 5)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def hhmm_to_minutes(series: pd.Series, column_name: str) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    parsed = pd.to_datetime(cleaned, format="%H:%M", errors="coerce")
    invalid = cleaned.notna() & parsed.isna()
    if invalid.any():
        examples = cleaned.loc[invalid].drop_duplicates().head(5).tolist()
        raise ValueError(f"{column_name}에 잘못된 HH:MM 값이 있습니다: {examples}")
    return parsed.dt.hour * 60 + parsed.dt.minute


def sortable_node(value: str) -> tuple[int, float | str]:
    text = str(value).strip()
    try:
        return 0, float(text)
    except ValueError:
        return 1, text


def prepare_od_time_windows(data: pd.DataFrame, time_bin: int) -> pd.DataFrame:
    status = data["final_status"].fillna("").astype("string").str.strip().str.casefold()
    data = data.loc[status.eq("complete")]
    if data.empty:
        raise ValueError("입력 파일에 final_status가 Complete인 행이 없습니다.")
    working = data[list(REQUIRED_COLUMNS)].copy()
    working["origin"] = working["origin"].astype("string").str.strip()
    working["destination"] = working["destination"].astype("string").str.strip()
    working["start"] = hhmm_to_minutes(
        working["time_window_start_hhmm"], "time_window_start_hhmm"
    )
    working["end"] = hhmm_to_minutes(
        working["time_window_end_hhmm"], "time_window_end_hhmm"
    )

    invalid = (
        working["origin"].isna()
        | working["origin"].eq("")
        | working["destination"].isna()
        | working["destination"].eq("")
        | working["start"].isna()
        | working["end"].isna()
    )
    if invalid.any():
        raise ValueError(f"OD 또는 time window가 잘못된 행이 있습니다: {working.index[invalid].tolist()[:10]}")
    if (working["end"] < working["start"]).any():
        rows = working.index[working["end"] < working["start"]].tolist()[:10]
        raise ValueError(f"종료 시각이 시작 시각보다 이른 행이 있습니다: {rows}")

    od_pairs = sorted(
        set(zip(working["origin"], working["destination"])),
        key=lambda pair: (sortable_node(pair[0]), sortable_node(pair[1])),
    )
    first_bin = int(working["start"].min() // time_bin * time_bin)
    last_edge = int(np.ceil(working["end"].max() / time_bin) * time_bin)
    time_bins = list(range(first_bin, last_edge, time_bin))
    od_index = {pair: index for index, pair in enumerate(od_pairs)}
    bin_index = {minute: index for index, minute in enumerate(time_bins)}
    values = np.zeros((len(od_pairs), len(time_bins)), dtype=int)

    for row in working.itertuples(index=False):
        # [start, end)와 실제로 겹치는 모든 bin에 이 요청을 한 번씩 더한다.
        first = int(row.start // time_bin * time_bin)
        final = int(np.ceil(row.end / time_bin) * time_bin)
        if final == first:  # 시작과 종료가 같은 예외적 요청도 한 bin에 표시
            final += time_bin
        od_row = od_index[(row.origin, row.destination)]
        for minute in range(first, final, time_bin):
            values[od_row, bin_index[minute]] += 1

    labels = [f"{origin} → {destination}" for origin, destination in od_pairs]
    return pd.DataFrame(values, index=labels, columns=time_bins)


def save_heatmap(
    frequency: pd.DataFrame,
    request_count: int,
    time_bin: int,
    output_path: Path,
    dpi: int,
) -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    values = frequency.to_numpy(dtype=int)
    maximum = max(1, int(values.max()))
    cmap = plt.colormaps["YlGnBu"].resampled(maximum + 1)
    cmap.set_under("white")
    norm = BoundaryNorm(np.arange(0.5, maximum + 1.5), cmap.N)

    fig_width = max(16.0, frequency.shape[1] * 0.16)
    fig_height = max(12.0, frequency.shape[0] * 0.31 + 2.8)
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
    ax.set_yticklabels(frequency.index, fontsize=7.5)
    ax.set_xlabel(f"시간대 ({time_bin}분 간격)")
    ax.set_ylabel("Origin → Destination")
    ax.set_title(
        "시간대별 Complete 요청 Time Window (OD별)\n"
        f"총 요청 {request_count:,}건 · 색과 숫자는 해당 시간대에 활성화된 time window 수",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    # 0이 아닌 셀에 건수를 표시하고, 배경이 진한 셀에는 흰 글자를 사용한다.
    for row, column in np.argwhere(values > 0):
        count = values[row, column]
        text_color = "white" if count >= max(2, maximum * 0.6) else "black"
        ax.text(column, row, str(count), ha="center", va="center", fontsize=7, color=text_color)

    ax.set_xticks(np.arange(-0.5, frequency.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, frequency.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, pad=0.012, ticks=np.arange(1, maximum + 1))
    colorbar.set_label("활성 Time Window 수")

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
        else input_path.with_name(f"{input_path.stem}_complete_time_windows_od.png")
    )
    if output_path.suffix.casefold() != ".png":
        raise ValueError("--output 경로의 확장자는 .png여야 합니다.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path, dtype=str, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise KeyError(f"입력 CSV에 필수 컬럼이 없습니다: {sorted(missing)}")

    frequency = prepare_od_time_windows(data, args.time_bin)
    complete_count = int(
        data["final_status"].fillna("").astype("string").str.strip().str.casefold().eq("complete").sum()
    )
    save_heatmap(frequency, complete_count, args.time_bin, output_path, args.dpi)
    print(f"Complete 요청: {complete_count:,}건")
    print(f"OD 조합: {frequency.shape[0]:,}개")
    print(f"PNG 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
