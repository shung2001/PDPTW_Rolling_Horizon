"""출발 노드별 전체 수요의 time window를 스펙트럼으로 시각화한다.

기준
----
* 노드: 요청의 출발 노드(origin)
* 시간: [time_window_start_hhmm, time_window_end_hhmm)와 겹치는 5분 구간
* 빈도: 각 노드·시간 구간에서 활성 상태인 전체 time window 수
* 대상: final_status와 무관한 모든 요청
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
REQUIRED_COLUMNS = {
    "origin",
    "time_window_start_hhmm",
    "time_window_end_hhmm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="출발 노드별 전체 time window 수요 스펙트럼을 생성합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="출력 PNG 경로 (기본값: 입력 CSV 폴더의 *_all_demand_by_node.png)",
    )
    parser.add_argument(
        "--time-bin",
        type=int,
        default=5,
        help="time window 집계 간격(분, 기본값: 5)",
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


def node_sort_key(value: str) -> tuple[int, float | str]:
    text = str(value).strip()
    try:
        return 0, float(text)
    except ValueError:
        return 1, text


def build_node_spectrum(data: pd.DataFrame, time_bin: int) -> pd.DataFrame:
    working = data[
        ["origin", "time_window_start_hhmm", "time_window_end_hhmm"]
    ].copy()
    working["origin"] = working["origin"].astype("string").str.strip()
    working["start"] = hhmm_to_minutes(
        working["time_window_start_hhmm"], "time_window_start_hhmm"
    )
    working["end"] = hhmm_to_minutes(
        working["time_window_end_hhmm"], "time_window_end_hhmm"
    )

    invalid = (
        working["origin"].isna()
        | working["origin"].eq("")
        | working["start"].isna()
        | working["end"].isna()
    )
    if invalid.any():
        rows = working.index[invalid].tolist()[:10]
        raise ValueError(f"노드 또는 time window가 잘못된 행이 있습니다: {rows}")
    if (working["end"] < working["start"]).any():
        rows = working.index[working["end"] < working["start"]].tolist()[:10]
        raise ValueError(f"종료 시각이 시작 시각보다 이른 행이 있습니다: {rows}")

    nodes = sorted(working["origin"].unique(), key=node_sort_key)
    first_bin = int(working["start"].min() // time_bin * time_bin)
    last_edge = int(np.ceil(working["end"].max() / time_bin) * time_bin)
    time_bins = np.arange(first_bin, last_edge, time_bin)
    node_index = {node: index for index, node in enumerate(nodes)}
    values = np.zeros((len(nodes), len(time_bins)), dtype=int)

    for row in working.itertuples(index=False):
        first = int(row.start // time_bin * time_bin)
        final = int(np.ceil(row.end / time_bin) * time_bin)
        if final == first:
            final += time_bin
        start_column = (first - first_bin) // time_bin
        end_column = (final - first_bin) // time_bin
        values[node_index[row.origin], start_column:end_column] += 1

    return pd.DataFrame(values, index=nodes, columns=time_bins)


def save_spectrum(
    spectrum: pd.DataFrame,
    request_count: int,
    time_bin: int,
    output_path: Path,
    dpi: int,
) -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    values = spectrum.to_numpy(dtype=int)
    maximum = max(1, int(values.max()))
    blues = plt.colormaps["YlGnBu"].resampled(maximum + 1)
    colors = [(1, 1, 1, 1)] + [blues(i) for i in range(1, maximum + 1)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, maximum + 1.5), cmap.N)

    fig, ax = plt.subplots(figsize=(18, max(6.0, len(spectrum.index) * 0.58)))
    image = ax.imshow(
        values,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    label_step = max(1, int(np.ceil(30 / time_bin)))
    tick_positions = np.arange(0, spectrum.shape[1], label_step)
    tick_minutes = spectrum.columns.to_numpy()[tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [f"{minute // 60:02d}:{minute % 60:02d}" for minute in tick_minutes],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(spectrum.shape[0]))
    ax.set_yticklabels([f"Node {node}" for node in spectrum.index])
    ax.set_xlabel(f"시간대 ({time_bin}분 간격)")
    ax.set_ylabel("출발 노드 (origin)")
    ax.set_title(
        "노드별 전체 수요 Time Window 스펙트럼\n"
        f"전체 요청 {request_count:,}건 · 색은 해당 시간대에 활성화된 time window 수",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    # 각 시간 구간의 빈도를 셀 위에 직접 표시한다.
    for row, column in np.argwhere(values > 0):
        count = values[row, column]
        text_color = "white" if count >= max(2, maximum * 0.6) else "black"
        ax.text(
            column,
            row,
            str(count),
            ha="center",
            va="center",
            fontsize=6.5,
            color=text_color,
        )

    ax.set_xticks(np.arange(-0.5, spectrum.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, spectrum.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(
        image, ax=ax, pad=0.015, ticks=np.arange(0, maximum + 1)
    )
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
        else input_path.with_name(f"{input_path.stem}_all_demand_by_node.png")
    )
    if output_path.suffix.casefold() != ".png":
        raise ValueError("--output 경로의 확장자는 .png여야 합니다.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path, dtype=str, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise KeyError(f"입력 CSV에 필수 컬럼이 없습니다: {sorted(missing)}")

    spectrum = build_node_spectrum(data, args.time_bin)
    save_spectrum(spectrum, len(data), args.time_bin, output_path, args.dpi)
    print(f"전체 요청: {len(data):,}건")
    print(f"출발 노드: {spectrum.shape[0]:,}개")
    print(f"PNG 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
