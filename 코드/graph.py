# -*- coding: utf-8 -*-
"""
penalty_per_vehicles.csv의 차량 대수별 penalty를 시각화한다.

최소 penalty와 해당 차량 대수를 그래프와 콘솔에 표시하고,
색상과 x축 눈금으로 차량 대수별 결과를 쉽게 구분할 수 있게 한다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = (
    BASE_DIR.parent
    / "자료"
    / "결과"
    / "Ortools"
    / "Penalty_per_vehicles"
    / "penalty_per_vehicles.csv"
)
OUTPUT_PATH = BASE_DIR / "penalty_per_vehicles.png"


def load_results(csv_path: Path) -> pd.DataFrame:
    """차량 대수와 penalty 컬럼을 검증하고 차량 대수 순으로 정렬한다."""
    df = pd.read_csv(csv_path)

    required_columns = {"num_vehicles", "total_penalty"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"필수 컬럼이 없습니다: {sorted(missing_columns)}")

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_rows = df[list(required_columns)].isna().any(axis=1)
    if invalid_rows.any():
        print(f"경고: 차량 대수 또는 penalty가 올바르지 않은 {invalid_rows.sum()}개 행을 제외합니다.")
        df = df.loc[~invalid_rows].copy()

    if df.empty:
        raise ValueError("그래프를 그릴 유효한 데이터가 없습니다.")

    df["num_vehicles"] = df["num_vehicles"].astype(int)
    return df.sort_values("num_vehicles").reset_index(drop=True)


def create_graph(df: pd.DataFrame, output_path: Path) -> tuple[float, list[int]]:
    """그래프를 생성·저장하고 최소 penalty와 해당 차량 대수를 반환한다."""
    min_penalty = float(df["total_penalty"].min())
    min_vehicles = (
        df.loc[df["total_penalty"].eq(min_penalty), "num_vehicles"]
        .astype(int)
        .tolist()
    )

    x = df["num_vehicles"]
    y = df["total_penalty"]

    fig, ax = plt.subplots(figsize=(12, 7))

    # 점의 색상에도 차량 대수를 반영해 어느 구간의 결과인지 바로 보이게 한다.
    ax.plot(x, y, color="#385f7c", linewidth=1.8, alpha=0.8, zorder=1)
    points = ax.scatter(
        x,
        y,
        c=x,
        cmap="viridis",
        s=42,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
        label="Penalty by vehicle count",
    )

    # 최소값이 여러 개인 경우 해당 차량 대수를 모두 강조한다.
    for vehicle_count in min_vehicles:
        ax.axvline(
            vehicle_count,
            color="#d1495b",
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
            zorder=0,
        )
    ax.axhline(
        min_penalty,
        color="#d1495b",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        zorder=0,
        label=f"Minimum penalty: {min_penalty:,.0f}",
    )
    ax.scatter(
        min_vehicles,
        [min_penalty] * len(min_vehicles),
        marker="*",
        s=260,
        color="#d1495b",
        edgecolors="white",
        linewidths=1.0,
        zorder=4,
        label="Minimum",
    )

    vehicle_text = ", ".join(map(str, min_vehicles))
    first_min_vehicle = min_vehicles[0]
    midpoint = (float(x.min()) + float(x.max())) / 2
    annotation_offset = (-155, 42) if first_min_vehicle >= midpoint else (28, 42)
    ax.annotate(
        f"Minimum\nVehicles: {vehicle_text}\nPenalty: {min_penalty:,.0f}",
        xy=(first_min_vehicle, min_penalty),
        xytext=annotation_offset,
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "#d1495b",
            "alpha": 0.95,
        },
        arrowprops={"arrowstyle": "->", "color": "#d1495b", "lw": 1.5},
        zorder=5,
    )

    ax.set_xlabel("Number of Vehicles", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Penalty", fontsize=11, fontweight="bold")
    ax.set_title(
        "Penalty by Number of Vehicles\n"
        f"Best result: {vehicle_text} vehicle(s), penalty {min_penalty:,.0f}",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    # 차량 대수를 정수 눈금으로 보여 주고 보조 눈금으로 구간 판독성을 높인다.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    ax.tick_params(axis="x", which="major", labelsize=9)
    ax.tick_params(axis="x", which="minor", length=3)

    colorbar = fig.colorbar(points, ax=ax, pad=0.02)
    colorbar.set_label("Number of Vehicles", fontweight="bold")
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))

    ax.margins(x=0.025, y=0.12)
    ax.grid(True, which="major", color="#b8c2cc", linewidth=0.8, alpha=0.45)
    ax.grid(True, which="minor", axis="x", color="#d8dee5", linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=True, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return min_penalty, min_vehicles


def main() -> None:
    df = load_results(CSV_PATH)
    min_penalty, min_vehicles = create_graph(df, OUTPUT_PATH)
    vehicle_text = ", ".join(map(str, min_vehicles))

    print(f"최소 penalty: {min_penalty:,.0f}")
    print(f"최소 penalty의 차량 대수: {vehicle_text}대")
    print(f"최적 결과: 차량 {vehicle_text}대 / penalty {min_penalty:,.0f}")
    print(f"그래프 저장 경로: {OUTPUT_PATH}")

    # GUI 백엔드에서는 화면에 보여 주고, 서버/ud14c스트 환경에서는 자원만 정리한다.
    if "agg" in plt.get_backend().lower():
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
