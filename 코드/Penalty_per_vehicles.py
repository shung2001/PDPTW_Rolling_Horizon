# -*- coding: utf-8 -*-
"""CONFIG에 지정한 차량 범위의 PDPTW 종합 drop penalty를 비교한다.

종합 penalty는 최종 상태가 Overdue인 batch의 final_drop_penalty 합계이다.
Complete batch는 실제로 drop되지 않았으므로 합계에서 제외한다.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd

import PDPTW_NEW as pdptw


CONFIG = {
    "min": 51,
    "max": 100,
}

SWEEP_OUTPUT_DIR = pdptw.OUTPUT_DIR / "Penalty_per_vehicles"
SUMMARY_PATH = SWEEP_OUTPUT_DIR / "penalty_per_vehicles.csv"
BEST_OUTPUT_DIR = SWEEP_OUTPUT_DIR / "best_penalty"
PDPTW_OUTPUT_FILENAMES = (
    "rolling_horizon_request_status.csv",
    "vehicle_route_legs.csv",
    "rolling_horizon_plan_history.csv",
    "rolling_horizon_summary.csv",
    "node_vehicle_occupancy_by_minute.csv",
    "vehicle_summary.csv",
)


def configured_vehicle_counts() -> range:
    min_vehicles = CONFIG["min"]
    max_vehicles = CONFIG["max"]
    if type(min_vehicles) is not int or type(max_vehicles) is not int:
        raise TypeError("CONFIG의 min과 max는 정수여야 합니다")
    if min_vehicles < 0 or max_vehicles < 0:
        raise ValueError("CONFIG의 min과 max는 0 이상이어야 합니다")
    if min_vehicles > max_vehicles:
        raise ValueError("CONFIG의 min은 max보다 클 수 없습니다")
    return range(min_vehicles, max_vehicles + 1)


def simulate_zero_vehicles(output_dir: Path) -> pd.DataFrame:
    """Solver 없이 기존 rolling-horizon 상태 갱신을 그대로 수행한다."""
    distance = pdptw.load_matrix(pdptw.DISTANCE_MATRIX_PATH, False)
    flight_time = pdptw.load_matrix(pdptw.TIME_MATRIX_PATH, True)
    nodes = pdptw.load_nodes(pdptw.NODE_REFERENCE_PATH)
    transportation_matrix = pdptw.load_named_matrix(pdptw.TRANSPORTATION_MATRIX_PATH, nodes)

    if set(distance.index) != set(flight_time.index) or not set(distance.index).issubset(nodes):
        raise ValueError("distance/time/node-reference의 node 집합이 일치하지 않습니다")

    tasks, batches = pdptw.load_tasks(
        pdptw.REQUEST_PATH,
        distance,
        flight_time,
        nodes,
        transportation_matrix,
    )
    maximum_max_wait = max(task.max_wait_min for task in tasks)
    simulation_end = (
        max(task.window_end for task in tasks)
        + int(flight_time.to_numpy().max())
        + pdptw.SERVICE_TIME_MINUTES
    )
    plans: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for horizon_start in range(
        0,
        simulation_end + 1,
        pdptw.REOPTIMIZATION_INTERVAL_MINUTES,
    ):
        pdptw.update_batch_states(batches, horizon_start)
        horizon_end = horizon_start + pdptw.ROLLING_HORIZON_MINUTES
        commit_end = horizon_start + pdptw.REOPTIMIZATION_INTERVAL_MINUTES
        active = [
            task
            for task in tasks
            if batches[task.batch_id].status == "Pending"
            and task.art <= horizon_end
            and task.window_end >= horizon_start
        ]
        active_batch_ids = {task.batch_id for task in active}
        for task in active:
            plans.append(
                {
                    "horizon_start": horizon_start,
                    "horizon_end": horizon_end,
                    "request_id": task.request_id,
                    "batch_id": task.batch_id,
                    "planned_visit": False,
                    "drop_penalty": pdptw.drop_penalty(
                        batches[task.batch_id],
                        maximum_max_wait,
                    ),
                }
            )
        pdptw.update_batch_states(
            batches,
            commit_end,
            increment_pending=True,
            attempted_batch_ids=active_batch_ids,
        )
        summaries.append(
            {
                "horizon_start": horizon_start,
                "horizon_end": horizon_end,
                "active_request_count": len(active),
                "planned_request_count": 0,
                "committed_request_count": 0,
                "completed_batch_count": 0,
                "pending_batch_count": sum(
                    batch.status == "Pending" for batch in batches.values()
                ),
                "overdue_batch_count": sum(
                    batch.status == "Overdue" for batch in batches.values()
                ),
                "objective_value": None,
                "solver_runtime_seconds": 0.0,
            }
        )

    pdptw.update_batch_states(
        batches,
        simulation_end + pdptw.ROLLING_HORIZON_MINUTES,
    )
    status_df = pdptw.batch_dataframe(batches, maximum_max_wait)
    route_df = pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_frames = {
        "rolling_horizon_request_status.csv": status_df,
        "vehicle_route_legs.csv": route_df,
        "rolling_horizon_plan_history.csv": pd.DataFrame(plans),
        "rolling_horizon_summary.csv": pd.DataFrame(summaries),
        "node_vehicle_occupancy_by_minute.csv": pdptw.occupancy_dataframe(
            route_df,
            nodes,
        ),
        "vehicle_summary.csv": pdptw.vehicle_summary_dataframe(route_df, []),
    }
    for filename, frame in output_frames.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
    return status_df


def run_vehicle_case(vehicle_count: int) -> tuple[pd.DataFrame, Path]:
    """지정한 차량 수로 시뮬레이션하고 최종 request 상태를 반환한다."""
    output_dir = SWEEP_OUTPUT_DIR / f"vehicles_{vehicle_count}"

    if vehicle_count == 0:
        return simulate_zero_vehicles(output_dir), output_dir

    pdptw.NUM_VEHICLES = vehicle_count
    pdptw.OUTPUT_DIR = output_dir
    pdptw.LOG_ROLLING_HORIZON = False
    pdptw.main()

    status_path = output_dir / "rolling_horizon_request_status.csv"
    return pd.read_csv(status_path, encoding="utf-8-sig"), output_dir


def summarize_penalty(
    vehicle_count: int,
    status_df: pd.DataFrame,
    runtime_seconds: float,
    output_dir: Path,
) -> dict[str, object]:
    overdue = status_df["final_status"].eq("Overdue")
    total_penalty = int(
        pd.to_numeric(
            status_df.loc[overdue, "final_drop_penalty"],
            errors="raise",
        ).sum()
    )
    return {
        "num_vehicles": vehicle_count,
        "total_penalty": total_penalty,
        "complete_batch_count": int(status_df["final_status"].eq("Complete").sum()),
        "overdue_batch_count": int(overdue.sum()),
        "total_batch_count": int(len(status_df)),
        "runtime_seconds": round(runtime_seconds, 3),
        "output_directory": str(output_dir),
    }


def publish_best_result(best_result: dict[str, object]) -> None:
    """최저 penalty 실행 결과를 PDPTW_NEW.py의 출력 파일명으로 복사한다."""
    source_dir = Path(str(best_result["output_directory"]))
    BEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in PDPTW_OUTPUT_FILENAMES:
        source = source_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"최저 penalty 결과 파일이 없습니다: {source}")
        shutil.copy2(source, BEST_OUTPUT_DIR / filename)


def main() -> None:
    SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    vehicle_counts = configured_vehicle_counts()
    print(
        f"차량 수 범위: CONFIG min={CONFIG['min']}, max={CONFIG['max']}",
        flush=True,
    )

    for vehicle_count in vehicle_counts:
        print(f"\n차량 {vehicle_count}대 계산 시작", flush=True)
        started = time.perf_counter()
        status_df, output_dir = run_vehicle_case(vehicle_count)
        result = summarize_penalty(
            vehicle_count,
            status_df,
            time.perf_counter() - started,
            output_dir,
        )
        results.append(result)

        # 각 차량 수의 계산이 끝날 때마다 저장하여 중간 결과도 보존한다.
        pd.DataFrame(results).to_csv(
            SUMMARY_PATH,
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"차량 {vehicle_count}대일 때 종합 penalty: "
            f"{int(result['total_penalty']):,} "
            f"(Complete={int(result['complete_batch_count']):,}, "
            f"Overdue={int(result['overdue_batch_count']):,})",
            flush=True,
        )

    best_result = min(
        results,
        key=lambda result: (
            int(result["total_penalty"]),
            int(result["num_vehicles"]),
        ),
    )
    publish_best_result(best_result)

    results_df = pd.DataFrame(results)
    results_df["is_best_penalty"] = (
        results_df["num_vehicles"] == int(best_result["num_vehicles"])
    )
    results_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print("\n차량 수별 종합 penalty", flush=True)
    print(
        results_df[
            [
                "num_vehicles",
                "total_penalty",
                "complete_batch_count",
                "overdue_batch_count",
                "is_best_penalty",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(
        f"\n최저 penalty: 차량 {int(best_result['num_vehicles'])}대, "
        f"종합 penalty={int(best_result['total_penalty']):,}",
        flush=True,
    )
    print(f"최저 penalty 상세 결과: {BEST_OUTPUT_DIR}", flush=True)
    print(f"\n요약 CSV: {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
