# -*- coding: utf-8 -*-
"""설정한 Revenue와 차량 수 범위의 PDPTW 종합 drop penalty를 비교한다.

종합 penalty는 최종 상태가 Overdue인 batch의 penalty 합계이다.
Complete batch는 실제로 drop되지 않았으므로 합계에서 제외한다.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd

import PDPTW_NEW_Remove_Depot_test as pdptw


VEHICLE_CONFIG = {
    "min": 50,
    "max": 150,
    "step": 5,
}

REVENUE_CONFIG = {
    "min": 50000,
    "max": 50000,
    "step": 10000,
}

# PDPTW_NEW_TEST의 OUTPUT_DIR 마지막 폴더는 기존 REVENUE 값이다.
# 그 상위 폴더를 sweep 루트로 사용해 <revenue>/vehicles_<count> 구조로 저장한다.
SWEEP_OUTPUT_DIR = pdptw.OUTPUT_DIR
PDPTW_OUTPUT_FILENAMES = (
    "rolling_horizon_request_status.csv",
    "vehicle_route_legs.csv",
    "rolling_horizon_plan_history.csv",
    "rolling_horizon_summary.csv",
    "node_vehicle_occupancy_by_minute.csv",
    "vehicle_summary.csv",
    "demand_summary.csv",
)


def configured_vehicle_counts() -> range:
    min_vehicles = VEHICLE_CONFIG["min"]
    max_vehicles = VEHICLE_CONFIG["max"]
    step_vehicles = VEHICLE_CONFIG["step"]
    if any(type(value) is not int for value in (min_vehicles, max_vehicles, step_vehicles)):
        raise TypeError("VEHICLE_CONFIG의 min, max, step은 정수여야 합니다")
    if min_vehicles < 0 or max_vehicles < 0:
        raise ValueError("VEHICLE_CONFIG의 min과 max는 0 이상이어야 합니다")
    if min_vehicles > max_vehicles:
        raise ValueError("VEHICLE_CONFIG의 min은 max보다 클 수 없습니다")
    if step_vehicles <= 0:
        raise ValueError("VEHICLE_CONFIG의 step은 1 이상이어야 합니다")
    return range(min_vehicles, max_vehicles + 1, step_vehicles)


def configured_revenues() -> range:
    min_revenue = REVENUE_CONFIG["min"]
    max_revenue = REVENUE_CONFIG["max"]
    step_revenue = REVENUE_CONFIG["step"]
    if any(type(value) is not int for value in (min_revenue, max_revenue, step_revenue)):
        raise TypeError("REVENUE_CONFIG의 min, max, step은 정수여야 합니다")
    if min_revenue < 0 or max_revenue < 0:
        raise ValueError("REVENUE_CONFIG의 min과 max는 0 이상이어야 합니다")
    if min_revenue > max_revenue:
        raise ValueError("REVENUE_CONFIG의 min은 max보다 클 수 없습니다")
    if step_revenue <= 0:
        raise ValueError("REVENUE_CONFIG의 step은 1 이상이어야 합니다")
    return range(min_revenue, max_revenue + 1, step_revenue)


def simulate_zero_vehicles(output_dir: Path) -> pd.DataFrame:
    """Solver 없이 기존 rolling-horizon 상태 갱신을 그대로 수행한다."""
    distance = pdptw.load_matrix(pdptw.DISTANCE_MATRIX_PATH, False)
    flight_time = pdptw.load_matrix(pdptw.TIME_MATRIX_PATH, True)
    nodes = pdptw.load_nodes(pdptw.NODE_REFERENCE_PATH)
    transportation_matrix = pdptw.load_named_matrix(pdptw.TRANSPORTATION_MATRIX_PATH, nodes)
    fare_matrix = pdptw.load_named_matrix(pdptw.TRANSPORTATION_MATRIX_COST, nodes)

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
                    "drop_penalty": pdptw.total_adddisjunction_cost(
                        batches[task.batch_id],
                        fare_matrix,
                        pdptw.REVENUE,
                        horizon_start,
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
    status_df = pdptw.batch_dataframe(batches, maximum_max_wait, fare_matrix)
    route_df = pd.DataFrame()
    total_passengers = sum(task.passengers for task in tasks)
    completed_passengers = sum(
        batch.alternatives[0].passengers
        for batch in batches.values()
        if batch.status == "Complete"
    )
    overdue_passengers = sum(
        batch.alternatives[0].passengers
        for batch in batches.values()
        if batch.status == "Overdue"
    )
    demand_summary_df = pd.DataFrame(
        [
            {
                "base_time": pdptw.BASE_TIME,
                "end_time": pdptw.END_TIME,
                "total_passengers": total_passengers,
                "total_batches": len(batches),
                "completed_passengers": completed_passengers,
                "overdue_passengers": overdue_passengers,
                "service_rate_percent": (
                    completed_passengers / total_passengers * 100
                    if total_passengers > 0
                    else 0
                ),
                "total_objectives": int(
                    pd.to_numeric(
                        status_df.loc[
                            status_df["final_status"].eq("Overdue"),
                            "final_cost(원)",
                        ],
                        errors="raise",
                    ).sum()
                ),
            }
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_frames = {
        "rolling_horizon_request_status.csv": status_df,
        "vehicle_route_legs.csv": route_df,
        "rolling_horizon_plan_history.csv": pd.DataFrame(plans),
        "rolling_horizon_summary.csv": pd.DataFrame(summaries),
        "node_vehicle_occupancy_by_minute.csv": pdptw.occupancy_dataframe(
            route_df,
            nodes,
            [],
        ),
        "vehicle_summary.csv": pd.DataFrame(),
        "demand_summary.csv": demand_summary_df,
    }
    for filename, frame in output_frames.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")
    return status_df


def run_vehicle_case(
    revenue: int,
    vehicle_count: int,
    revenue_output_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    """지정한 차량 수로 시뮬레이션하고 최종 request 상태를 반환한다."""
    output_dir = revenue_output_dir / f"vehicles_{vehicle_count}"

    pdptw.REVENUE = revenue

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
    if "final_drop_penalty" in status_df.columns:
        penalty_column = "final_drop_penalty"
    elif "final_cost(원)" in status_df.columns:
        penalty_column = "final_cost(원)"
    else:
        raise KeyError(
            "rolling_horizon_request_status.csv에 penalty 열이 없습니다: "
            "final_drop_penalty 또는 final_cost(원) 열이 필요합니다"
        )
    total_penalty = int(
        pd.to_numeric(
            status_df.loc[overdue, penalty_column],
            errors="raise",
        ).sum()
    )
    demand_summary_path = output_dir / "demand_summary.csv"
    demand_summary_df = pd.read_csv(demand_summary_path, encoding="utf-8-sig")
    if len(demand_summary_df) != 1 or "total_objectives" not in demand_summary_df.columns:
        raise ValueError(
            f"demand_summary.csv에는 total_objectives 열을 포함한 한 행이 필요합니다: "
            f"{demand_summary_path}"
        )
    total_objectives = int(
        pd.to_numeric(demand_summary_df["total_objectives"], errors="raise").iloc[0]
    )
    return {
        "revenue": pdptw.REVENUE,
        "num_vehicles": vehicle_count,
        "total_penalty": total_penalty,
        "total_objectives": total_objectives,
        "complete_batch_count": int(status_df["final_status"].eq("Complete").sum()),
        "overdue_batch_count": int(overdue.sum()),
        "total_batch_count": int(len(status_df)),
        "runtime_seconds": round(runtime_seconds, 3),
    }


def publish_best_result(
    best_result: dict[str, object],
    best_output_dir: Path,
) -> None:
    """최저 penalty 실행 결과를 PDPTW_NEW.py의 출력 파일명으로 복사한다."""
    source_dir = Path(str(best_result["output_directory"]))
    best_output_dir.mkdir(parents=True, exist_ok=True)
    for filename in PDPTW_OUTPUT_FILENAMES:
        source = source_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"최저 penalty 결과 파일이 없습니다: {source}")
        shutil.copy2(source, best_output_dir / filename)


def main() -> None:
    SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vehicle_counts = tuple(configured_vehicle_counts())
    revenues = tuple(configured_revenues())
    print(
        f"Revenue 범위: REVENUE_CONFIG min={REVENUE_CONFIG['min']:,}, "
        f"max={REVENUE_CONFIG['max']:,}, step={REVENUE_CONFIG['step']:,}",
        flush=True,
    )
    print(
        f"차량 수 범위: VEHICLE_CONFIG min={VEHICLE_CONFIG['min']}, "
        f"max={VEHICLE_CONFIG['max']}, step={VEHICLE_CONFIG['step']}",
        flush=True,
    )

    for revenue in revenues:
        revenue_output_dir = SWEEP_OUTPUT_DIR / str(revenue)
        summary_path = revenue_output_dir / "penalty_per_vehicles.csv"
        best_output_dir = revenue_output_dir / "best_penalty"
        revenue_output_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, object]] = []

        print(f"\nRevenue {revenue:,}원 계산 시작", flush=True)
        for vehicle_count in vehicle_counts:
            print(f"\nRevenue {revenue:,}원, 차량 {vehicle_count}대 계산 시작", flush=True)
            started = time.perf_counter()
            status_df, output_dir = run_vehicle_case(
                revenue,
                vehicle_count,
                revenue_output_dir,
            )
            result = summarize_penalty(
                vehicle_count,
                status_df,
                time.perf_counter() - started,
                output_dir,
            )
            results.append(result)

            # 각 차량 수의 계산이 끝날 때마다 저장하여 중간 결과도 보존한다.
            pd.DataFrame(results).to_csv(
                summary_path,
                index=False,
                encoding="utf-8-sig",
            )
            print(
                f"Revenue {revenue:,}원, 차량 {vehicle_count}대일 때 종합 penalty: "
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
        publish_best_result(best_result, best_output_dir)

        best_total_objectives_result = min(
            results,
            key=lambda result: (
                int(result["total_objectives"]),
                int(result["num_vehicles"]),
            ),
        )

        results_df = pd.DataFrame(results)
        results_df["is_best_penalty"] = (
            results_df["num_vehicles"] == int(best_result["num_vehicles"])
        )
        results_df["is_best_total_objectives"] = (
            results_df["num_vehicles"]
            == int(best_total_objectives_result["num_vehicles"])
        )
        results_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        print(f"\nRevenue {revenue:,}원 차량 수별 종합 penalty", flush=True)
        print(
            results_df[
                [
                    "num_vehicles",
                    "total_penalty",
                    "total_objectives",
                    "complete_batch_count",
                    "overdue_batch_count",
                    "is_best_penalty",
                    "is_best_total_objectives",
                ]
            ].to_string(index=False),
            flush=True,
        )
        print(
            f"\nRevenue {revenue:,}원 최저 penalty: "
            f"차량 {int(best_result['num_vehicles'])}대, "
            f"종합 penalty={int(best_result['total_penalty']):,}",
            flush=True,
        )
        print(
            f"Revenue {revenue:,}원 최저 total_objectives: "
            f"차량 {int(best_total_objectives_result['num_vehicles'])}대, "
            f"total_objectives={int(best_total_objectives_result['total_objectives']):,}",
            flush=True,
        )
        print(f"최저 penalty 상세 결과: {best_output_dir}", flush=True)
        print(f"요약 CSV: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
