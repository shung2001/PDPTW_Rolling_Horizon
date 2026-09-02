"""Overdue 요청을 추출하고 시간 구간을 하나의 스펙트럼으로 시각화한다.

가로축은 시각이며 색은 해당 시각에 겹치는 Overdue time window의 개수다.
겹치는 요청이 많을수록 진한 색으로 표시된다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # GUI가 없는 환경에서도 이미지 저장 가능
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


PDPTW_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    PDPTW_ROOT / "자료" / "결과" / "Ortools" / "d5000"
    / "Penalty_Per_Vehicles" / "Original" / "130000" / "vehicles_150"
    / "rolling_horizon_request_status.csv"
)
REQUIRED_COLUMNS = {
    "final_status", "time_window_start_hhmm", "time_window_end_hhmm"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overdue 행을 추출하고 시간 구간을 단일 스펙트럼으로 시각화합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="결과 저장 폴더 (기본값: 입력 CSV와 같은 폴더)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def hhmm_to_minutes(series: pd.Series, column_name: str) -> pd.Series:
    """HH:MM 문자열을 자정 이후 경과 분으로 변환한다."""
    cleaned = series.astype("string").str.strip()
    parsed = pd.to_datetime(cleaned, format="%H:%M", errors="coerce")
    invalid = cleaned.notna() & parsed.isna()
    if invalid.any():
        examples = cleaned.loc[invalid].drop_duplicates().head(5).tolist()
        raise ValueError(f"{column_name}에 잘못된 HH:MM 값이 있습니다: {examples}")
    return parsed.dt.hour * 60 + parsed.dt.minute


def make_overlap_spectrum(
    starts: pd.Series, ends: pd.Series
) -> tuple[np.ndarray, int, int]:
    """분 단위 시간축에서 동시에 겹치는 Overdue 구간 수를 계산한다."""
    valid = starts.notna() & ends.notna()
    starts = starts.loc[valid].astype(int)
    ends = ends.loc[valid].astype(int)
    if starts.empty:
        return np.zeros(1, dtype=int), 0, 1
    if (ends < starts).any():
        raise ValueError("종료 시각이 시작 시각보다 이른 time window가 있습니다.")

    axis_start, axis_end = int(starts.min()), int(ends.max())
    differences = np.zeros(axis_end - axis_start + 2, dtype=int)
    for start, end in zip(starts, ends):
        differences[start - axis_start] += 1
        differences[end - axis_start + 1] -= 1
    return np.cumsum(differences[:-1]), axis_start, axis_end


def save_spectrum(
    spectrum: np.ndarray,
    axis_start: int,
    axis_end: int,
    request_count: int,
    output_path: Path,
    dpi: int,
) -> None:
    """중첩도를 하나의 가로 스펙트럼 이미지로 저장한다."""
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    max_overlap = max(1, int(spectrum.max()))
    reds = plt.colormaps["YlOrRd"].resampled(max_overlap + 1)
    colors = [(1, 1, 1, 1)] + [reds(i) for i in range(1, max_overlap + 1)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, max_overlap + 1.5), cmap.N)

    fig, ax = plt.subplots(figsize=(16, 3.8))
    image = ax.imshow(
        spectrum[np.newaxis, :],
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=[axis_start, axis_end + 1, 0, 1],
    )
    ax.set_yticks([])
    ax.set_xlim(axis_start, axis_end + 1)
    ax.set_xlabel("시각")
    ax.set_title(
        "Overdue Time-window Spectrum\n"
        f"Overdue {request_count:,}건",
        fontsize=14, fontweight="bold", pad=14,
    )

    tick_interval = 30 if axis_end - axis_start <= 12 * 60 else 60
    first_tick = ((axis_start + tick_interval - 1) // tick_interval) * tick_interval
    ticks = np.arange(first_tick, axis_end + 1, tick_interval)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{minute // 60:02d}:{minute % 60:02d}" for minute in ticks],
        rotation=45,
    )
    ax.grid(axis="x", color="black", alpha=0.15, linewidth=0.7)

    colorbar = fig.colorbar(
        image, ax=ax, orientation="horizontal", pad=0.38, fraction=0.16,
        ticks=np.arange(max_overlap + 1),
    )
    colorbar.set_label("동시에 겹치는 Overdue 구간 수")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"입력 CSV를 찾을 수 없습니다: {input_path}")
    if args.dpi <= 0:
        raise ValueError("--dpi는 1 이상의 정수여야 합니다.")

    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else input_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_path, dtype=str, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise KeyError(f"입력 CSV에 필수 컬럼이 없습니다: {sorted(missing)}")

    mask = data["final_status"].fillna("").str.strip().str.casefold() == "overdue"
    overdue = data.loc[mask].copy()
    overdue["_start_minutes"] = hhmm_to_minutes(
        overdue["time_window_start_hhmm"], "time_window_start_hhmm"
    )
    overdue["_end_minutes"] = hhmm_to_minutes(
        overdue["time_window_end_hhmm"], "time_window_end_hhmm"
    )
    overdue = overdue.sort_values(
        ["_start_minutes", "_end_minutes"], kind="stable", na_position="last"
    )

    stem = input_path.stem
    overdue_path = output_dir / f"{stem}_overdue.csv"
    spectrum_path = output_dir / f"{stem}_overdue_spectrum.png"
    overdue[data.columns].to_csv(overdue_path, index=False, encoding="utf-8-sig")

    spectrum, axis_start, axis_end = make_overlap_spectrum(
        overdue["_start_minutes"], overdue["_end_minutes"]
    )
    save_spectrum(
        spectrum, axis_start, axis_end, len(overdue), spectrum_path, args.dpi
    )

    print(f"전체 행: {len(data):,}건")
    print(f"Overdue 행: {len(overdue):,}건")
    print(f"Overdue CSV: {overdue_path}")
    print(f"단일 스펙트럼 이미지: {spectrum_path}")


if __name__ == "__main__":
    main()
