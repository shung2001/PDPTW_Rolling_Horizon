"""Plot overdue batches and penalty for each vehicle count in one single run.

The default input is ``.../single/130000/Original/add_6mins``. Each numeric
subdirectory contains a ``rolling_horizon_request_status.csv`` from one vehicle
count. The two PNG files are saved in the input directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

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
    / "VP_기존"
    / "Rolling_Horizon_구간_30"
    / "d1000"
    / "single"
    / "130000"
    / "Original"
    / "add_6mins"
)
STATUS_NAME = "rolling_horizon_request_status.csv"


def load_vehicle_results(data_dir: Path) -> pd.DataFrame:
    """Read final request status for every numeric vehicle-count folder."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    vehicle_dirs = sorted(
        (path for path in data_dir.iterdir() if path.is_dir() and path.name.isdecimal()),
        key=lambda path: int(path.name),
    )
    if not vehicle_dirs:
        raise FileNotFoundError(f"No vehicle-count folders found in: {data_dir}")

    results = []
    for vehicle_dir in vehicle_dirs:
        status_path = vehicle_dir / STATUS_NAME
        if not status_path.is_file():
            raise FileNotFoundError(f"Missing status CSV: {status_path}")

        status = pd.read_csv(status_path, encoding="utf-8-sig")
        if status.empty or "final_status" not in status.columns:
            raise ValueError(f"Missing final_status values: {status_path}")

        penalty_column = next(
            (name for name in ("final_drop_penalty", "final_cost(원)") if name in status),
            None,
        )
        if penalty_column is None:
            raise ValueError(f"Missing penalty column in: {status_path}")

        overdue = status["final_status"].eq("Overdue")
        penalties = pd.to_numeric(status.loc[overdue, penalty_column], errors="raise")
        if penalties.isna().any() or not np.isfinite(penalties).all():
            raise ValueError(f"Invalid overdue penalty in: {status_path}")

        results.append(
            {
                "num_vehicles": int(vehicle_dir.name),
                "total_penalty": penalties.sum(),
                "overdue_batch_count": int(overdue.sum()),
            }
        )

    return pd.DataFrame(results)


def save_chart(
    results: pd.DataFrame, column: str, ylabel: str, output_path: Path, dpi: int
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.plot(
        results["num_vehicles"],
        results[column],
        color="#1f77b4",
        marker="o",
        markersize=5,
        linewidth=1.8,
    )
    ax.set_xticks(results["num_vehicles"].tolist())
    ax.set_xlabel("Number of Vehicles", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel} by Number of Vehicles", fontsize=16, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def generate_charts(data_dir: Path, dpi: int = 200) -> list[Path]:
    if dpi <= 0:
        raise ValueError(f"dpi must be positive: {dpi}")
    data_dir = data_dir.resolve()
    results = load_vehicle_results(data_dir)

    outputs = [
        ("total_penalty", "Penalty", data_dir / "penalty_by_num_vehicles.png"),
        (
            "overdue_batch_count",
            "Number of Overdue Batches",
            data_dir / "overdue_by_num_vehicles.png",
        ),
    ]
    for column, ylabel, output_path in outputs:
        save_chart(results, column, ylabel, output_path, dpi)
        print(f"created: {output_path}")
    print(f"vehicle_counts={','.join(map(str, results['num_vehicles']))}")
    return [output_path for _, _, output_path in outputs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    generate_charts(args.data_dir, args.dpi)


if __name__ == "__main__":
    main()
