# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Rolling-Horizon Dynamic Electric PDPTW - NVIDIA cuOpt Routing version
====================================================================

목표
----
기존 OR-Tools RoutingModel 기반 구현의 운영 의미를 최대한 유지하면서,
Horizon 내부의 경로 탐색을 NVIDIA cuOpt Routing으로 전환한다.

핵심 설계
---------
1) 각 RequestTask(batch)는 하나의 원자적 OD 운항이다.
   - cuOpt 내부에서는 Pickup + Delivery pair로 표현한다.
   - Solver capacity에는 실제 탑승객 수가 아니라 VEHICLE_CAPACITY를 사용한다.
     이는 한 batch를 pickup한 뒤 반드시 해당 delivery를 바로 수행하도록 만드는
     'atomic flight lock' 역할을 한다. 실제 탑승객 수는 출력/통계에는 그대로 사용한다.

2) Physical_Node_matrix 구조를 유지한다.
   - cost/time matrix는 물리 location 수 x 물리 location 수이다.
   - 동일 물리 위치에 여러 시간대/수요 order가 존재해도 matrix를 복제하지 않는다.
   - set_order_locations()로 order -> physical location을 매핑한다.

3) Rolling Horizon / 상태 전이는 기존 목적을 유지한다.
   - pending/planned/onboard/completed/overdue
   - 실제 Pickup Departure가 허용시간 내 완료되면 Visit=True
   - 미래 계획은 다음 Horizon에서 재최적화한다.

4) Drop penalty 목적함수를 기존 코드와 동일하게 유지한다.
   - 수요·거리·Joby/지상교통 시간차를 모두 request_drop_penalty에 포함한다.
   - pickup/delivery 양쪽에 요청 prize의 절반씩을 배분하면,
     cuOpt의 (route_cost - collected_prize) 목적은 상수항을 제외하고
     OR-Tools의 (route_cost + dropped_drop_penalty)와 동등해진다.

5) IMPORTANT - 배터리/충전
   cuOpt Routing Python API는 OR-Tools의 사용자 정의 RemainingRange Dimension처럼
   "arc distance만큼 감소 + waiting/service 동안 연속 충전 + threshold에서 강제 충전"
   을 직접 표현하는 사용자 정의 누적 resource API를 제공하지 않는다.

   따라서:
   - 경로 탐색 자체는 cuOpt GPU Routing이 담당한다.
   - 실제 commit 직전에 배터리/충전 규칙을 정확히 다시 계산한다.
   - 첫 commit task가 배터리/시간상 불가능하면 해당 task-vehicle 조합을 금지하여
     제한 횟수 내에서 cuOpt를 재호출하는 repair loop를 사용한다.
   - 실제 실행 상태(VehicleState)는 이 검증을 통과한 운항만 반영한다.

이 방식은 "GPU가 행렬만 계산"하던 CuPy 버전과 달리,
실제 routing.Solve() 경로 탐색을 cuOpt가 GPU에서 수행한다.
"""

import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_CUOPT_IMPORT_ERROR: Exception | None = None
try:
    import cudf
    from cuopt import routing
except Exception as exc:
    cudf = None
    routing = None
    _CUOPT_IMPORT_ERROR = exc

try:
    import cupy as cp
except Exception:
    cp = None


# ============================================================
# 사용자 설정
# ============================================================
# cuOpt Python SDK의 Windows 공식 사용 형태는 WSL2이므로,
# 아래 Windows 경로는 WSL 실행 시 자동으로 /mnt/c/... 로 변환한다.
WINDOWS_INPUT_DIR = (
    r"C:\Users\c\Desktop\대학생활\학연생\새로운_경로방식\자료\심화\교수님_자료\finalDemand_1"
)
WINDOWS_OUTPUT_DIR = (
    r"C:\Users\c\Desktop\대학생활\학연생\새로운_경로방식\자료\심화\교수님_자료\결과_cuOpt"
)


def is_wsl() -> bool:
    if os.name != "posix":
        return False
    try:
        text = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
        return "microsoft" in text or "wsl" in text
    except OSError:
        return False


def runtime_path(windows_path: str) -> Path:
    if not is_wsl():
        return Path(windows_path)

    match = re.match(r"^([A-Za-z]):\\(.*)$", windows_path)
    if match is None:
        return Path(windows_path)
    drive = match.group(1).lower()
    rest = match.group(2).replace("\\", "/")
    return Path(f"/mnt/{drive}/{rest}")


INPUT_DIR = runtime_path(WINDOWS_INPUT_DIR)
OUTPUT_DIR = runtime_path(WINDOWS_OUTPUT_DIR)

REQUEST_PATH = INPUT_DIR / "d5000_remove.csv"
DISTANCE_MATRIX_PATH = INPUT_DIR / "Distance_matrix_1_9.csv"
TIME_MATRIX_PATH = INPUT_DIR / "Time_matrix_1_9.csv"
NODE_REFERENCE_PATH = INPUT_DIR / "vp_reference_remove_동탄.csv"

BASE_TIME = "05:40"
NUM_VEHICLES = 303
VEHICLE_CAPACITY = 3
DEPOT_ROUTE_NODE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

ROLLING_HORIZON_MINUTES = 30
REOPTIMIZATION_INTERVAL_MINUTES = 30
TIME_LIMIT_SECONDS = 0.5
SERVICE_TIME_MINUTES = 5
REQUEST_TAKE_OFF_TIME_MINUTES = 2
ARRIVAL_BUFFER_MINUTES = 3
MIN_OD_DISTANCE_KM = 0.0

DROP_PENALTY_BASE = 1_000_000
DEMAND_PENALTY_WEIGHT = 100_000
DISTANCE_PENALTY_WEIGHT = 10_000
TIME_PENALTY_WEIGHT = 1_000
MIN_REQUEST_DROP_PENALTY = 1

INITIAL_REMAINING_RANGE_KM = 160.0
MAX_REMAINING_RANGE_KM = 160.0
MIN_REMAINING_RANGE_KM = 15.0
LOW_RANGE_CHARGE_TARGET_KM = 100.0
CHARGING_RATE_KM_PER_MIN = 4.267

# cuOpt 설정
CUOPT_VERBOSE = False
CUOPT_BATTERY_REPAIR_MAX_RESOLVES = 0

# None이면 NUM_VEHICLES는 '사용 가능한 fleet의 상한'으로만 동작한다.
# 즉 모든 차량을 매 Horizon마다 강제로 운항시키지 않는다.
# 정수 K를 넣을 때만 cuOpt에 최소 사용 차량 수 K를 요청한다.
CUOPT_MIN_VEHICLES: int | None = None

# 거리 목적함수 단위. 기존 OR-Tools 코드와 맞추기 위해 km -> m cost로 사용한다.
DISTANCE_COST_SCALE = 1000.0


COLUMN_ALIASES = {
    "origin": ("n", "origin", "origin_node", "출발노드", "pickup_node"),
    "destination": ("m", "destination", "destination_node", "도착노드", "delivery_node"),
    "passengers": ("cnt", "passenger", "passengers", "demand", "수요"),
    "max_wait": ("maxWait_min_round", "maxWait_min", "max_wait_min"),
    "departure": ("depature_time", "departure_time", "requested_departure_time"),
    "ground_time": ("movement_time", "ground_movement_time", "car_time"),
    "distance": ("distance", "distance_km"),
    "joby_time": ("Joby_Flight_Time", "joby_flight_time", "uam_time"),
    "route_node": ("vp_id", "route_node_id", "matrix_node_id", "node_id"),
    "physical_node": ("Node", "physical_node_id", "physical_id"),
    "address": ("vpname", "physical_address", "address", "주소"),
    "latitude": ("lat", "latitude", "Y_4326"),
    "longitude": ("lon", "longitude", "X_4326"),
}


# ============================================================
# 데이터 구조
# ============================================================
@dataclass(frozen=True)
class NodeInfo:
    route_node_id: str
    physical_node_id: str
    address: str
    longitude: float
    latitude: float


@dataclass
class RequestTask:
    task_key: str
    request_id: str
    batch_id: int
    origin: str
    destination: str
    passengers: int
    desired_departure: int
    min_available_departure: int
    max_available_departure: int
    ground_movement_time_min: int
    joby_flight_time_min: int
    distance_km: float
    drop_penalty: int
    time_performance_penalty: int
    status: str = "pending"
    visit: bool | None = None
    assigned_vehicle_id: int | None = None
    actual_departure_time: int | None = None
    actual_arrival_time: int | None = None
    completion_time: int | None = None
    drop_penalty_applied: bool = False


@dataclass
class VehicleState:
    vehicle_id: int
    route_node_id: str
    node_arrival_time: int
    available_time: int
    remaining_range_km: float


@dataclass
class CuOptMatrixStore:
    node_ids: tuple[str, ...]
    node_to_index: dict[str, int]
    distance_km_cpu: np.ndarray
    distance_cost_cpu: np.ndarray
    time_cpu: np.ndarray
    cost_gpu: Any
    time_gpu: Any
    maximum_flight_time: int
    device_name: str

    def distance(self, from_node: str, to_node: str) -> float:
        return float(
            self.distance_km_cpu[
                self.node_to_index[from_node],
                self.node_to_index[to_node],
            ]
        )

    def travel_time(self, from_node: str, to_node: str) -> int:
        return int(
            self.time_cpu[
                self.node_to_index[from_node],
                self.node_to_index[to_node],
            ]
        )


@dataclass
class CuOptHorizonModel:
    data_model: Any
    active_tasks: list[RequestTask]
    task_by_pickup_order: dict[int, RequestTask]
    task_by_delivery_order: dict[int, RequestTask]
    pickup_order_by_task_key: dict[str, int]
    delivery_order_by_task_key: dict[str, int]
    model_end: int


@dataclass
class BatteryExecutionPlan:
    vehicle_id: int
    task_key: str
    empty_departure: int
    empty_arrival: int
    passenger_departure: int
    passenger_arrival: int
    pickup_wait: int
    pre_reposition_charge_time: int
    pickup_charge_time: int
    delivery_charge_time: int
    mandatory_pickup_charge_wait: int
    mandatory_delivery_charge_wait: int
    empty_distance_km: float
    range_before_empty_km: float
    range_after_empty_km: float
    departure_range_km: float
    range_after_passenger_km: float
    range_after_delivery_service_km: float


# ============================================================
# 입력 처리
# ============================================================
def normalize_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def find_column(df: pd.DataFrame, key: str, required: bool = True) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in df.columns}
    for candidate in COLUMN_ALIASES[key]:
        matched = normalized.get(candidate.lower())
        if matched is not None:
            return matched
    if required:
        raise KeyError(f"'{key}'에 해당하는 열을 찾지 못했습니다. 후보: {COLUMN_ALIASES[key]}")
    return None


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"파일 인코딩을 확인해 주세요: {path}")


def load_matrix(path: Path, integer: bool) -> pd.DataFrame:
    df = read_table(path)
    first_column = df.columns[0]
    if str(first_column).lower().startswith("unnamed") or len(df.columns) == len(df) + 1:
        df = df.set_index(first_column)
    elif df.index.equals(pd.RangeIndex(len(df))):
        df = df.set_index(first_column)

    df.index = [normalize_id(value) for value in df.index]
    df.columns = [normalize_id(value) for value in df.columns]
    df = df.apply(pd.to_numeric, errors="raise")
    if set(df.index) != set(df.columns):
        raise ValueError(f"행렬의 행·열 노드가 일치하지 않습니다: {path}")
    df = df.loc[df.index, df.index]
    return df.astype(int) if integer else df.astype(float)


def load_nodes(path: Path) -> dict[str, NodeInfo]:
    df = read_table(path)
    route_col = find_column(df, "route_node")
    physical_col = find_column(df, "physical_node", required=False) or route_col
    address_col = find_column(df, "address", required=False)
    lon_col = find_column(df, "longitude")
    lat_col = find_column(df, "latitude")

    nodes: dict[str, NodeInfo] = {}
    for _, row in df.iterrows():
        route_id = normalize_id(row[route_col])
        nodes[route_id] = NodeInfo(
            route_node_id=route_id,
            physical_node_id=normalize_id(row[physical_col]),
            address="" if address_col is None else str(row[address_col]).strip(),
            longitude=float(row[lon_col]),
            latitude=float(row[lat_col]),
        )
    return nodes


def ensure_cuopt_available() -> None:
    if routing is None or cudf is None:
        message = (
            "NVIDIA cuOpt/cudf를 불러오지 못했습니다. cuOpt Python SDK는 Linux 또는 "
            "Windows 11 + WSL2 환경에서 CUDA와 함께 설치해야 합니다."
        )
        if _CUOPT_IMPORT_ERROR is None:
            raise ImportError(message)
        raise RuntimeError(message) from _CUOPT_IMPORT_ERROR


def build_cuopt_matrix_store(
    distance_matrix: pd.DataFrame,
    time_matrix: pd.DataFrame,
) -> CuOptMatrixStore:
    ensure_cuopt_available()

    if list(distance_matrix.index) != list(time_matrix.index):
        time_matrix = time_matrix.loc[distance_matrix.index, distance_matrix.columns]

    node_ids = tuple(str(value) for value in distance_matrix.index)
    node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}

    distance_km_cpu = np.ascontiguousarray(
        distance_matrix.loc[node_ids, node_ids].to_numpy(dtype=np.float64)
    )
    distance_cost_cpu = np.ascontiguousarray(
        distance_km_cpu * DISTANCE_COST_SCALE,
        dtype=np.float32,
    )
    time_cpu = np.ascontiguousarray(
        time_matrix.loc[node_ids, node_ids].to_numpy(dtype=np.int32)
    )

    # cudf DataFrame 생성 시 데이터가 GPU 메모리로 올라간다.
    cost_gpu = cudf.DataFrame(distance_cost_cpu)
    time_gpu = cudf.DataFrame(time_cpu.astype(np.float32))

    device_name = "CUDA GPU"
    if cp is not None:
        try:
            props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            raw_name = props.get("name", b"CUDA GPU")
            device_name = raw_name.decode(errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
        except Exception:
            pass

    return CuOptMatrixStore(
        node_ids=node_ids,
        node_to_index=node_to_index,
        distance_km_cpu=distance_km_cpu,
        distance_cost_cpu=distance_cost_cpu,
        time_cpu=time_cpu,
        cost_gpu=cost_gpu,
        time_gpu=time_gpu,
        maximum_flight_time=int(time_cpu.max()),
        device_name=device_name,
    )


def base_time_minutes() -> int:
    hour, minute = [int(value) for value in BASE_TIME.split(":")[:2]]
    return hour * 60 + minute


def to_offset_minutes(value: object) -> int:
    if pd.isna(value):
        raise ValueError("출발시각이 비어 있습니다.")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(round(float(value)))

    text = str(value).strip()
    if text.replace(".", "", 1).isdigit():
        return int(round(float(text)))

    parts = text.split(":")
    if len(parts) < 2:
        raise ValueError(f"시각을 해석할 수 없습니다: {value}")
    absolute = int(parts[0]) * 60 + int(parts[1])
    offset = absolute - base_time_minutes()
    if offset < 0:
        offset += 24 * 60
    return offset


def compute_request_drop_penalty(
    passengers: int,
    distance_km: float,
    joby_time: int,
    ground_time: int,
) -> int:
    # OR-Tools 원본과 동일하게 시간 성능항까지 미수행 penalty에 포함한다.
    value = (
        DROP_PENALTY_BASE
        + DEMAND_PENALTY_WEIGHT * passengers
        + DISTANCE_PENALTY_WEIGHT * distance_km
        + TIME_PENALTY_WEIGHT * (joby_time - ground_time)
    )
    return max(MIN_REQUEST_DROP_PENALTY, int(round(value)))


def load_tasks(
    request_path: Path,
    matrices: CuOptMatrixStore,
    nodes: dict[str, NodeInfo],
) -> list[RequestTask]:
    raw = read_table(request_path)
    origin_col = find_column(raw, "origin")
    destination_col = find_column(raw, "destination")
    passenger_col = find_column(raw, "passengers")
    max_wait_col = find_column(raw, "max_wait")
    departure_col = find_column(raw, "departure", required=False)
    ground_col = find_column(raw, "ground_time")

    standardized: list[dict[str, Any]] = []
    for row_index, row in raw.iterrows():
        origin = normalize_id(row[origin_col])
        destination = normalize_id(row[destination_col])
        if origin == destination:
            continue
        if origin not in nodes or destination not in nodes:
            raise KeyError(f"수요 row={row_index}의 노드가 노드 참조표에 없습니다: {origin}->{destination}")
        if origin not in matrices.node_to_index or destination not in matrices.node_to_index:
            raise KeyError(f"수요 row={row_index}의 노드가 거리/시간행렬에 없습니다: {origin}->{destination}")

        if departure_col is not None:
            desired_departure = to_offset_minutes(row[departure_col])
        elif {"depT_hr", "depT_m"}.issubset(raw.columns):
            desired_departure = int(row["depT_hr"]) * 60 + int(row["depT_m"]) - base_time_minutes()
        else:
            raise KeyError("출발시각 열 또는 depT_hr/depT_m 열이 필요합니다.")

        standardized.append(
            {
                "origin": origin,
                "destination": destination,
                "desired_departure": int(desired_departure),
                "max_wait": max(0, int(math.ceil(float(row[max_wait_col])))),
                "passengers": max(1, int(math.ceil(float(row[passenger_col])))),
                "ground_time": int(math.ceil(float(row[ground_col]))),
            }
        )

    if not standardized:
        return []

    grouped = (
        pd.DataFrame(standardized)
        .groupby(
            ["origin", "destination", "desired_departure", "max_wait", "ground_time"],
            as_index=False,
        )
        .agg(passengers=("passengers", "sum"))
        .sort_values(["desired_departure", "origin", "destination"])
        .reset_index(drop=True)
    )

    tasks: list[RequestTask] = []
    for _, row in grouped.iterrows():
        origin = str(row["origin"])
        destination = str(row["destination"])
        desired = int(row["desired_departure"])
        max_wait = int(row["max_wait"])
        distance = matrices.distance(origin, destination)
        joby_time = matrices.travel_time(origin, destination)
        if distance < MIN_OD_DISTANCE_KM:
            continue

        total_passengers = int(row["passengers"])
        request_id = f"{origin}_{destination}_{desired}"
        batch_id = 1
        remaining = total_passengers
        while remaining > 0:
            batch_passengers = min(VEHICLE_CAPACITY, remaining)
            task_key = f"{request_id}_B{batch_id}"
            tasks.append(
                RequestTask(
                    task_key=task_key,
                    request_id=request_id,
                    batch_id=batch_id,
                    origin=origin,
                    destination=destination,
                    passengers=batch_passengers,
                    desired_departure=desired,
                    min_available_departure=max(0, desired - max_wait),
                    max_available_departure=desired + max_wait,
                    ground_movement_time_min=int(row["ground_time"]),
                    joby_flight_time_min=joby_time,
                    distance_km=distance,
                    drop_penalty=compute_request_drop_penalty(
                        batch_passengers,
                        distance,
                        joby_time,
                        int(row["ground_time"]),
                    ),
                    time_performance_penalty=0,
                )
            )
            remaining -= batch_passengers
            batch_id += 1

    return tasks


# ============================================================
# cuOpt Horizon 모델
# ============================================================
def effective_service_prize(task: RequestTask) -> float:
    # OR-Tools의 route_cost + dropped_drop_penalty는 전체 penalty 상수항을 제외하면
    # cuOpt의 route_cost - collected_prize와 동일하다.
    return float(task.drop_penalty)


def build_cuopt_horizon_model(
    active_tasks: list[RequestTask],
    vehicle_states: list[VehicleState],
    matrices: CuOptMatrixStore,
    horizon_start: int,
    horizon_end: int,
    forbidden_vehicle_by_task: dict[str, set[int]] | None = None,
) -> CuOptHorizonModel:
    ensure_cuopt_available()
    forbidden_vehicle_by_task = forbidden_vehicle_by_task or {}

    vehicle_count = len(vehicle_states)
    pair_count = len(active_tasks)
    order_count = pair_count * 2

    model_end = (
        horizon_end
        + matrices.maximum_flight_time
        + SERVICE_TIME_MINUTES
        + REQUEST_TAKE_OFF_TIME_MINUTES
        + ARRIVAL_BUFFER_MINUTES
    )

    data_model = routing.DataModel(
        len(matrices.node_ids),
        vehicle_count,
        order_count,
    )
    data_model.add_cost_matrix(matrices.cost_gpu)
    data_model.add_transit_time_matrix(matrices.time_gpu)

    pickup_locations = [matrices.node_to_index[task.origin] for task in active_tasks]
    delivery_locations = [matrices.node_to_index[task.destination] for task in active_tasks]
    data_model.set_order_locations(
        cudf.Series(pickup_locations + delivery_locations, dtype="int32")
    )

    pickup_orders = list(range(pair_count))
    delivery_orders = list(range(pair_count, pair_count * 2))
    data_model.set_pickup_delivery_pairs(
        cudf.Series(pickup_orders, dtype="int32"),
        cudf.Series(delivery_orders, dtype="int32"),
    )

    # 원자적 OD batch 보존:
    # 실제 승객이 1~2명이어도 solver capacity에서는 한 batch가 기체 전 좌석을 점유하도록 두어
    # 다른 pickup이 해당 passenger flight 중간에 끼어들지 못하게 한다.
    capacity_demand = [VEHICLE_CAPACITY] * pair_count + [-VEHICLE_CAPACITY] * pair_count
    vehicle_capacity = [VEHICLE_CAPACITY] * vehicle_count
    data_model.add_capacity_dimension(
        "atomic_od_lock",
        cudf.Series(capacity_demand, dtype="int32"),
        cudf.Series(vehicle_capacity, dtype="int32"),
    )

    # actual_departure = pickup_service_start + SERVICE + TAKEOFF
    pickup_service_total = SERVICE_TIME_MINUTES + REQUEST_TAKE_OFF_TIME_MINUTES
    pickup_earliest = [
        max(0, task.min_available_departure - pickup_service_total)
        for task in active_tasks
    ]
    pickup_latest = [
        max(0, task.max_available_departure - pickup_service_total)
        for task in active_tasks
    ]
    delivery_earliest = [0] * pair_count
    delivery_latest = [model_end] * pair_count

    data_model.set_order_time_windows(
        cudf.Series(pickup_earliest + delivery_earliest, dtype="int32"),
        cudf.Series(pickup_latest + delivery_latest, dtype="int32"),
    )
    data_model.set_order_service_times(
        cudf.Series(
            [pickup_service_total] * pair_count + [SERVICE_TIME_MINUTES] * pair_count,
            dtype="int32",
        )
    )

    # vehicle의 현재 물리 위치와 가용시각을 Rolling Horizon 상태에서 이어받는다.
    start_locations = [matrices.node_to_index[state.route_node_id] for state in vehicle_states]
    # Horizon 내부에서는 복귀를 강제하지 않으므로 return location은 start와 동일하게 두고
    # 마지막 return trip을 drop한다.
    data_model.set_vehicle_locations(
        cudf.Series(start_locations, dtype="int32"),
        cudf.Series(start_locations, dtype="int32"),
    )
    data_model.set_drop_return_trips(cudf.Series([True] * vehicle_count, dtype="bool"))

    vehicle_earliest = [max(horizon_start, int(state.available_time)) for state in vehicle_states]
    vehicle_latest = [model_end] * vehicle_count
    data_model.set_vehicle_time_windows(
        cudf.Series(vehicle_earliest, dtype="int32"),
        cudf.Series(vehicle_latest, dtype="int32"),
    )

    # cuOpt는 pickup-delivery pair 양쪽 order에 prize가 있어야 pair를 선택한다.
    # pair 전체 prize를 절반씩 배분하면 OR-Tools의 요청별 drop penalty와 총액이 같다.
    pair_prizes = [effective_service_prize(task) / 2.0 for task in active_tasks]
    prizes = pair_prizes + pair_prizes
    data_model.set_order_prizes(cudf.Series(prizes, dtype="float32"))
    data_model.set_objective_function(
        cudf.Series([routing.Objective.PRIZE, routing.Objective.COST]),
        cudf.Series([1.0, 1.0], dtype="float32"),
    )

    if CUOPT_MIN_VEHICLES is not None:
        requested_min = int(CUOPT_MIN_VEHICLES)
        if requested_min < 0 or requested_min > vehicle_count:
            raise ValueError("CUOPT_MIN_VEHICLES는 0 이상 NUM_VEHICLES 이하여야 합니다.")
        if requested_min > 0:
            data_model.set_min_vehicles(min(requested_min, pair_count))

    task_by_pickup_order: dict[int, RequestTask] = {}
    task_by_delivery_order: dict[int, RequestTask] = {}
    pickup_order_by_task_key: dict[str, int] = {}
    delivery_order_by_task_key: dict[str, int] = {}

    all_vehicle_ids = set(range(vehicle_count))
    for i, task in enumerate(active_tasks):
        pickup_id = i
        delivery_id = pair_count + i
        task_by_pickup_order[pickup_id] = task
        task_by_delivery_order[delivery_id] = task
        pickup_order_by_task_key[task.task_key] = pickup_id
        delivery_order_by_task_key[task.task_key] = delivery_id

        forbidden = forbidden_vehicle_by_task.get(task.task_key, set())
        if forbidden:
            allowed = sorted(all_vehicle_ids - forbidden)
            if allowed:
                allowed_series = cudf.Series(allowed, dtype="int32")
                data_model.add_order_vehicle_match(pickup_id, allowed_series)
                data_model.add_order_vehicle_match(delivery_id, allowed_series)
            else:
                # 모든 차량이 금지된 경우 prize를 0으로 만드는 것만으로는 강제 drop을 보장하지 못하므로
                # 이 상황은 repair loop에서 더 이상 재시도하지 않도록 상위 로직이 처리한다.
                pass

    return CuOptHorizonModel(
        data_model=data_model,
        active_tasks=active_tasks,
        task_by_pickup_order=task_by_pickup_order,
        task_by_delivery_order=task_by_delivery_order,
        pickup_order_by_task_key=pickup_order_by_task_key,
        delivery_order_by_task_key=delivery_order_by_task_key,
        model_end=model_end,
    )


def solve_cuopt_horizon(model: CuOptHorizonModel):
    settings = routing.SolverSettings()
    settings.set_time_limit(float(TIME_LIMIT_SECONDS))
    settings.set_verbose_mode(bool(CUOPT_VERBOSE))
    return routing.Solve(model.data_model, settings)


def solution_route_to_pandas(solution: Any) -> pd.DataFrame:
    if solution is None:
        return pd.DataFrame()
    try:
        route_gpu = solution.get_route()
    except Exception:
        return pd.DataFrame()
    if route_gpu is None or len(route_gpu) == 0:
        return pd.DataFrame()
    return route_gpu.to_pandas().reset_index(drop=True)


def extract_cuopt_plan(
    model: CuOptHorizonModel,
    solution: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], pd.DataFrame]:
    task_plan: dict[str, dict[str, Any]] = {
        task.task_key: {
            "planned_visit": False,
            "vehicle_id": None,
            "planned_departure": None,
            "planned_arrival": None,
        }
        for task in model.active_tasks
    }
    route_plan: list[dict[str, Any]] = []
    route_df = solution_route_to_pandas(solution)
    if route_df.empty:
        return route_plan, task_plan, route_df

    pickup_rows_by_task: dict[str, dict[str, Any]] = {}
    delivery_rows_by_task: dict[str, dict[str, Any]] = {}

    has_type = "type" in route_df.columns
    for row_position, row in route_df.iterrows():
        row_type = str(row["type"]).strip().lower() if has_type else ""
        if row_type == "depot":
            continue

        try:
            order_id = int(row["route"])
        except (TypeError, ValueError):
            continue

        if row_type == "pickup" or order_id in model.task_by_pickup_order:
            task = model.task_by_pickup_order.get(order_id)
            if task is not None:
                pickup_rows_by_task[task.task_key] = {
                    "vehicle_id": int(row["truck_id"]),
                    "arrival_stamp": int(round(float(row["arrival_stamp"]))),
                    "row_position": int(row_position),
                    "order_id": order_id,
                }
        elif row_type == "delivery" or order_id in model.task_by_delivery_order:
            task = model.task_by_delivery_order.get(order_id)
            if task is not None:
                delivery_rows_by_task[task.task_key] = {
                    "vehicle_id": int(row["truck_id"]),
                    "arrival_stamp": int(round(float(row["arrival_stamp"]))),
                    "row_position": int(row_position),
                    "order_id": order_id,
                }

    pickup_service_total = SERVICE_TIME_MINUTES + REQUEST_TAKE_OFF_TIME_MINUTES
    per_vehicle: dict[int, list[tuple[int, RequestTask, dict[str, Any], dict[str, Any]]]] = {}

    for task in model.active_tasks:
        pickup = pickup_rows_by_task.get(task.task_key)
        delivery = delivery_rows_by_task.get(task.task_key)
        if pickup is None or delivery is None:
            continue
        if pickup["vehicle_id"] != delivery["vehicle_id"]:
            # cuOpt PDP에서는 원래 발생하면 안 되지만 방어적으로 무시한다.
            continue

        vehicle_id = int(pickup["vehicle_id"])
        planned_departure = int(pickup["arrival_stamp"] + pickup_service_total)
        planned_arrival = int(delivery["arrival_stamp"])

        task_plan[task.task_key] = {
            "planned_visit": True,
            "vehicle_id": vehicle_id,
            "planned_departure": planned_departure,
            "planned_arrival": planned_arrival,
        }
        per_vehicle.setdefault(vehicle_id, []).append(
            (pickup["row_position"], task, pickup, delivery)
        )

    for vehicle_id, items in per_vehicle.items():
        for sequence, (_, task, pickup, delivery) in enumerate(sorted(items, key=lambda x: x[0])):
            route_plan.append(
                {
                    "vehicle_id": vehicle_id,
                    "sequence": sequence,
                    "task_key": task.task_key,
                    "planned_departure": int(pickup["arrival_stamp"] + pickup_service_total),
                    "planned_arrival": int(delivery["arrival_stamp"]),
                }
            )

    return route_plan, task_plan, route_df


def compute_planned_route_cost_m(route_df: pd.DataFrame, matrices: CuOptMatrixStore) -> int:
    if route_df.empty:
        return 0

    total = 0.0
    for _, vehicle_rows in route_df.groupby("truck_id", sort=False):
        rows = vehicle_rows.reset_index(drop=True)
        if len(rows) <= 1:
            continue
        for i in range(len(rows) - 1):
            current = rows.iloc[i]
            nxt = rows.iloc[i + 1]
            # set_drop_return_trips(True)이므로 마지막 Depot 복귀 arc는 목적함수에서 제외한다.
            if i + 1 == len(rows) - 1 and str(nxt.get("type", "")).lower() == "depot":
                continue
            from_loc = int(current["location"])
            to_loc = int(nxt["location"])
            total += float(matrices.distance_cost_cpu[from_loc, to_loc])
    return int(round(total))


def compute_equivalent_horizon_objective(
    active_tasks: list[RequestTask],
    task_plan: dict[str, dict[str, Any]],
    route_cost_m: int,
) -> tuple[int, int, int]:
    dropped_penalty = 0
    served_time_penalty = 0
    for task in active_tasks:
        if task_plan[task.task_key]["planned_visit"]:
            served_time_penalty += task.time_performance_penalty
        else:
            dropped_penalty += task.drop_penalty
    objective = int(route_cost_m + dropped_penalty + served_time_penalty)
    return objective, int(dropped_penalty), int(served_time_penalty)


# ============================================================
# 배터리/충전: 실제 commit 전 정확한 상태 계산
# ============================================================
def ceil_charge_minutes(required_range_km: float) -> int:
    if required_range_km <= 1e-9:
        return 0
    return int(math.ceil(required_range_km / CHARGING_RATE_KM_PER_MIN))


def plan_battery_feasible_execution(
    state: VehicleState,
    task: RequestTask,
    planned_departure: int,
    matrices: CuOptMatrixStore,
) -> BatteryExecutionPlan | None:
    vehicle_id = state.vehicle_id
    empty_distance = matrices.distance(state.route_node_id, task.origin)
    empty_time = matrices.travel_time(state.route_node_id, task.origin)

    # 공차 이동만으로 최대 Range를 초과하면 어떤 충전으로도 불가능하다.
    if empty_distance + MIN_REMAINING_RANGE_KM > MAX_REMAINING_RANGE_KM + 1e-9:
        return None
    if task.distance_km + MIN_REMAINING_RANGE_KM > MAX_REMAINING_RANGE_KM + 1e-9:
        return None

    current_range = min(MAX_REMAINING_RANGE_KM, max(0.0, state.remaining_range_km))

    # 현재 노드에서 공차 이동 후 최소 15km를 남기기 위해 필요한 선충전.
    needed_precharge = max(
        0.0,
        MIN_REMAINING_RANGE_KM + empty_distance - current_range,
    )
    precharge_time = ceil_charge_minutes(needed_precharge)
    range_before_empty = min(
        MAX_REMAINING_RANGE_KM,
        current_range + CHARGING_RATE_KM_PER_MIN * precharge_time,
    )
    empty_departure = int(state.available_time) + precharge_time
    empty_arrival = empty_departure + empty_time
    range_after_empty = range_before_empty - empty_distance

    if range_after_empty < MIN_REMAINING_RANGE_KM - 1e-6:
        return None

    base_departure_without_wait = (
        empty_arrival + SERVICE_TIME_MINUTES + REQUEST_TAKE_OFF_TIME_MINUTES
    )

    # 1) cuOpt가 계획한 실제 departure보다 빨리 출발하지 않는다.
    planned_wait = max(0, int(planned_departure) - base_departure_without_wait)
    # 2) 요청 최소 출발시각을 만족한다.
    window_wait = max(0, task.min_available_departure - base_departure_without_wait)

    # 3) 배터리 규칙을 만족하기 위해 pickup에서 필요한 총 충전량을 계산한다.
    departure_target_range = task.distance_km + MIN_REMAINING_RANGE_KM
    mandatory_low_range = range_after_empty <= MIN_REMAINING_RANGE_KM + 1e-9
    if mandatory_low_range:
        departure_target_range = max(departure_target_range, LOW_RANGE_CHARGE_TARGET_KM)
    departure_target_range = min(MAX_REMAINING_RANGE_KM, departure_target_range)

    required_pickup_charge_minutes = ceil_charge_minutes(
        max(0.0, departure_target_range - range_after_empty)
    )
    battery_extra_wait = max(0, required_pickup_charge_minutes - SERVICE_TIME_MINUTES)

    pickup_wait = max(planned_wait, window_wait, battery_extra_wait)
    passenger_departure = (
        empty_arrival
        + pickup_wait
        + SERVICE_TIME_MINUTES
        + REQUEST_TAKE_OFF_TIME_MINUTES
    )

    if passenger_departure > task.max_available_departure:
        return None

    pickup_charge_time = pickup_wait + SERVICE_TIME_MINUTES
    departure_range = min(
        MAX_REMAINING_RANGE_KM,
        range_after_empty + CHARGING_RATE_KM_PER_MIN * pickup_charge_time,
    )

    if departure_range < task.distance_km + MIN_REMAINING_RANGE_KM - 1e-6:
        return None
    if mandatory_low_range and departure_range < min(LOW_RANGE_CHARGE_TARGET_KM, MAX_REMAINING_RANGE_KM) - 1e-6:
        return None

    passenger_arrival = passenger_departure + task.joby_flight_time_min
    range_after_passenger = departure_range - task.distance_km
    if range_after_passenger < MIN_REMAINING_RANGE_KM - 1e-6:
        return None

    delivery_charge_time = SERVICE_TIME_MINUTES
    if range_after_passenger <= MIN_REMAINING_RANGE_KM + 1e-9:
        required_total = ceil_charge_minutes(
            max(0.0, LOW_RANGE_CHARGE_TARGET_KM - range_after_passenger)
        )
        delivery_charge_time = max(SERVICE_TIME_MINUTES, required_total)

    range_after_delivery_service = min(
        MAX_REMAINING_RANGE_KM,
        range_after_passenger + CHARGING_RATE_KM_PER_MIN * delivery_charge_time,
    )

    mandatory_pickup_charge_wait = max(
        0,
        required_pickup_charge_minutes - SERVICE_TIME_MINUTES,
    ) if mandatory_low_range else 0
    mandatory_delivery_charge_wait = max(0, delivery_charge_time - SERVICE_TIME_MINUTES)

    return BatteryExecutionPlan(
        vehicle_id=vehicle_id,
        task_key=task.task_key,
        empty_departure=empty_departure,
        empty_arrival=empty_arrival,
        passenger_departure=passenger_departure,
        passenger_arrival=passenger_arrival,
        pickup_wait=pickup_wait,
        pre_reposition_charge_time=precharge_time,
        pickup_charge_time=pickup_charge_time,
        delivery_charge_time=delivery_charge_time,
        mandatory_pickup_charge_wait=mandatory_pickup_charge_wait,
        mandatory_delivery_charge_wait=mandatory_delivery_charge_wait,
        empty_distance_km=empty_distance,
        range_before_empty_km=range_before_empty,
        range_after_empty_km=range_after_empty,
        departure_range_km=departure_range,
        range_after_passenger_km=range_after_passenger,
        range_after_delivery_service_km=range_after_delivery_service,
    )


def committable_tasks_by_vehicle(
    route_plan: list[dict[str, Any]],
    horizon_start: int,
    commit_end: int,
) -> dict[int, list[dict[str, Any]]]:
    """commit 구간에 계획된 모든 task를 차량별 운항 순서대로 반환한다."""
    committable: dict[int, list[dict[str, Any]]] = {}
    for row in sorted(route_plan, key=lambda item: (item["vehicle_id"], item["sequence"])):
        departure = int(row["planned_departure"])
        if horizon_start <= departure < commit_end:
            vehicle_id = int(row["vehicle_id"])
            committable.setdefault(vehicle_id, []).append(row)
    return committable


def copied_vehicle_state(state: VehicleState) -> VehicleState:
    """battery repair 검증용으로 실제 상태를 손상하지 않는 복사본을 만든다."""
    return VehicleState(
        vehicle_id=state.vehicle_id,
        route_node_id=state.route_node_id,
        node_arrival_time=state.node_arrival_time,
        available_time=state.available_time,
        remaining_range_km=state.remaining_range_km,
    )


def apply_execution_to_vehicle_state(
    state: VehicleState,
    task: RequestTask,
    execution: BatteryExecutionPlan,
) -> None:
    """한 운항이 끝난 뒤 다음 운항 검증/실행에 사용할 VehicleState를 갱신한다."""
    state.route_node_id = task.destination
    state.node_arrival_time = execution.passenger_arrival
    state.available_time = execution.passenger_arrival + execution.delivery_charge_time
    state.remaining_range_km = execution.range_after_delivery_service_km


def solve_horizon_with_battery_repair(
    active_tasks: list[RequestTask],
    vehicle_states: list[VehicleState],
    matrices: CuOptMatrixStore,
    horizon_start: int,
    horizon_end: int,
    commit_end: int,
) -> tuple[
    CuOptHorizonModel | None,
    Any,
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    pd.DataFrame,
    float,
    int,
]:
    forbidden: dict[str, set[int]] = {}
    total_solver_seconds = 0.0
    last_model: CuOptHorizonModel | None = None
    last_solution: Any = None
    last_route_plan: list[dict[str, Any]] = []
    last_task_plan: dict[str, dict[str, Any]] = {
        task.task_key: {
            "planned_visit": False,
            "vehicle_id": None,
            "planned_departure": None,
            "planned_arrival": None,
        }
        for task in active_tasks
    }
    last_route_df = pd.DataFrame()

    task_lookup = {task.task_key: task for task in active_tasks}
    state_by_vehicle = {state.vehicle_id: state for state in vehicle_states}

    max_attempts = max(1, CUOPT_BATTERY_REPAIR_MAX_RESOLVES + 1)
    for attempt in range(max_attempts):
        model = build_cuopt_horizon_model(
            active_tasks,
            vehicle_states,
            matrices,
            horizon_start,
            horizon_end,
            forbidden_vehicle_by_task=forbidden,
        )

        solve_start = time.perf_counter()
        solution = solve_cuopt_horizon(model)
        total_solver_seconds += time.perf_counter() - solve_start

        route_plan, task_plan, route_df = extract_cuopt_plan(model, solution)

        last_model = model
        last_solution = solution
        last_route_plan = route_plan
        last_task_plan = task_plan
        last_route_df = route_df

        failures: list[tuple[str, int]] = []
        committable_by_vehicle = committable_tasks_by_vehicle(
            route_plan,
            horizon_start,
            commit_end,
        )

        # 실제 VehicleState는 건드리지 않고 차량별 복사본으로 commit 구간 전체를 순차 검증한다.
        # 앞 task의 도착 위치/가용시각/배터리 상태를 뒤 task 검증에 그대로 인계한다.
        for vehicle_id, plans in committable_by_vehicle.items():
            temp_state = copied_vehicle_state(state_by_vehicle[vehicle_id])

            for plan in plans:
                task = task_lookup[plan["task_key"]]
                execution = plan_battery_feasible_execution(
                    temp_state,
                    task,
                    int(plan["planned_departure"]),
                    matrices,
                )

                if execution is None:
                    failures.append((task.task_key, vehicle_id))
                    # 이 task를 수행하지 못하면 이후 cuOpt route의 연속성이 깨지므로
                    # 해당 차량의 뒤 task 검증은 재최적화 이후 다시 수행한다.
                    break

                # cuOpt 계획상 commit 구간이었더라도 앞선 실제 충전/운항으로 밀려
                # 실제 출발이 commit_end 이후가 되면 이번 구간에서는 확정하지 않는다.
                # 이는 infeasible이 아니므로 vehicle-task 금지 사유로 처리하지 않는다.
                if execution.passenger_departure >= commit_end:
                    break

                apply_execution_to_vehicle_state(temp_state, task, execution)

        if not failures:
            return (
                model,
                solution,
                route_plan,
                task_plan,
                route_df,
                total_solver_seconds,
                attempt,
            )

        if attempt >= max_attempts - 1:
            break

        changed = False
        for task_key, vehicle_id in failures:
            current = forbidden.setdefault(task_key, set())
            if vehicle_id not in current and len(current) < len(vehicle_states) - 1:
                current.add(vehicle_id)
                changed = True
        if not changed:
            break

    return (
        last_model,
        last_solution,
        last_route_plan,
        last_task_plan,
        last_route_df,
        total_solver_seconds,
        max_attempts - 1,
    )


# ============================================================
# 확정 구간 실행 및 상태 인계
# ============================================================
def format_hhmm(offset_minutes: int | float | None) -> str:
    if offset_minutes is None or pd.isna(offset_minutes):
        return ""
    absolute = base_time_minutes() + int(round(float(offset_minutes)))
    return f"{(absolute // 60) % 24:02d}:{absolute % 60:02d}"


def add_stop_interval(
    rows: list[dict[str, Any]],
    vehicle_id: int,
    route_node_id: str,
    start: int,
    end: int,
) -> None:
    if end <= start:
        return
    rows.append(
        {
            "vehicle_id": vehicle_id,
            "route_node_id": route_node_id,
            "start_time_min": int(start),
            "end_time_min": int(end),
        }
    )


def event_row(
    *,
    vehicle_id: int,
    sequence: int,
    task: RequestTask,
    event_type: str,
    movement_type: str,
    from_node: str,
    to_node: str,
    departure_time: int,
    arrival_time: int,
    waiting_time: int,
    service_time: int,
    passengers: int,
    range_at_departure: float,
    range_at_arrival: float,
    charged_range: float,
    flight_distance_km: float,
    nodes: dict[str, NodeInfo],
    charge_remain: float | None = None,
) -> dict[str, Any]:
    origin = nodes[from_node]
    destination = nodes[to_node]
    return {
        "vehicle_id": vehicle_id,
        "route_sequence": sequence,
        "request_id": task.request_id,
        "batch_id": task.batch_id,
        "task_key": task.task_key,
        "event_type": event_type,
        "movement_type": movement_type,
        "departure_route_node_id": from_node,
        "arrival_route_node_id": to_node,
        "departure_physical_node_id": origin.physical_node_id,
        "arrival_physical_node_id": destination.physical_node_id,
        "departure_address": origin.address,
        "arrival_address": destination.address,
        "departure_X_4326": origin.longitude,
        "departure_Y_4326": origin.latitude,
        "arrival_X_4326": destination.longitude,
        "arrival_Y_4326": destination.latitude,
        "arrival_time_min": arrival_time,
        "arrival_time_hhmm": format_hhmm(arrival_time),
        "departure_time_min": departure_time,
        "departure_time_hhmm": format_hhmm(departure_time),
        "min_available_departure": task.min_available_departure,
        "max_available_departure": task.max_available_departure,
        "waiting_time_min": waiting_time,
        "service_time_min": service_time,
        "takeoff_time_min": REQUEST_TAKE_OFF_TIME_MINUTES if event_type == "Pickup" else 0,
        "passenger_load": passengers,
        "remaining_range_at_departure_km": round(range_at_departure, 3),
        "remaining_range_at_arrival_km": round(range_at_arrival, 3),
        "charged_range_km": round(charged_range, 3),
        "flight_distance_km": round(flight_distance_km, 3),
        "flight_time_min": max(0, int(arrival_time) - int(departure_time)),
        "charging_time_min": 0,
        "mandatory_charge_wait_min": 0,
        "remaining_range_after_service_km": round(
            range_at_arrival if charge_remain is None else charge_remain, 3
        ),
        "Charge_remain": round(
            range_at_arrival if charge_remain is None else charge_remain, 3
        ),
    }


def commit_all_committable_tasks(
    route_plan: list[dict[str, Any]],
    task_lookup: dict[str, RequestTask],
    vehicle_states: list[VehicleState],
    matrices: CuOptMatrixStore,
    nodes: dict[str, NodeInfo],
    horizon_start: int,
    commit_end: int,
    leg_rows: list[dict[str, Any]],
    stop_rows: list[dict[str, Any]],
) -> int:
    """commit 구간 안에서 실제 수행 가능한 모든 task를 차량별 순서대로 확정한다."""
    state_by_vehicle = {state.vehicle_id: state for state in vehicle_states}
    committable_by_vehicle = committable_tasks_by_vehicle(
        route_plan,
        horizon_start,
        commit_end,
    )

    committed = 0

    for vehicle_id, plans in committable_by_vehicle.items():
        state = state_by_vehicle[vehicle_id]

        for plan in plans:
            task = task_lookup[plan["task_key"]]
            if task.status not in {"pending", "planned"}:
                continue

            execution = plan_battery_feasible_execution(
                state,
                task,
                int(plan["planned_departure"]),
                matrices,
            )

            if execution is None:
                # repair loop 이후에도 불가능한 경우 이 운항부터 뒤쪽 route의 연속성을
                # 신뢰할 수 없으므로 해당 차량의 이번 commit을 여기서 중단한다.
                break

            # 계획은 commit 구간에 있었더라도 실제 앞선 운항/충전 때문에 출발이
            # commit_end 밖으로 밀리면 다음 재최적화에서 다시 판단한다.
            if execution.passenger_departure >= commit_end:
                break

            add_stop_interval(
                stop_rows,
                vehicle_id,
                state.route_node_id,
                state.node_arrival_time,
                execution.empty_departure,
            )

            if execution.empty_distance_km > 0:
                movement_row = event_row(
                    vehicle_id=vehicle_id,
                    sequence=len(leg_rows),
                    task=task,
                    event_type="Movement",
                    movement_type="empty_repositioning",
                    from_node=state.route_node_id,
                    to_node=task.origin,
                    departure_time=execution.empty_departure,
                    arrival_time=execution.empty_arrival,
                    waiting_time=0,
                    service_time=0,
                    passengers=0,
                    range_at_departure=execution.range_before_empty_km,
                    range_at_arrival=execution.range_after_empty_km,
                    charged_range=max(
                        0.0,
                        execution.range_before_empty_km - state.remaining_range_km,
                    ),
                    flight_distance_km=execution.empty_distance_km,
                    nodes=nodes,
                )
                movement_row["charging_time_min"] = execution.pre_reposition_charge_time
                movement_row["mandatory_charge_wait_min"] = execution.pre_reposition_charge_time
                leg_rows.append(movement_row)

            add_stop_interval(
                stop_rows,
                vehicle_id,
                task.origin,
                execution.empty_arrival,
                execution.passenger_departure,
            )

            pickup_charge = max(
                0.0,
                execution.departure_range_km - execution.range_after_empty_km,
            )
            pickup_row = event_row(
                vehicle_id=vehicle_id,
                sequence=len(leg_rows),
                task=task,
                event_type="Pickup",
                movement_type="pickup_service",
                from_node=task.origin,
                to_node=task.origin,
                departure_time=execution.passenger_departure,
                arrival_time=execution.empty_arrival,
                waiting_time=execution.pickup_wait,
                service_time=SERVICE_TIME_MINUTES,
                passengers=task.passengers,
                range_at_departure=execution.departure_range_km,
                range_at_arrival=execution.range_after_empty_km,
                charged_range=pickup_charge,
                flight_distance_km=0.0,
                nodes=nodes,
                charge_remain=execution.departure_range_km,
            )
            pickup_row["charging_time_min"] = execution.pickup_charge_time
            pickup_row["mandatory_charge_wait_min"] = execution.mandatory_pickup_charge_wait
            leg_rows.append(pickup_row)

            delivery_charge = max(
                0.0,
                execution.range_after_delivery_service_km - execution.range_after_passenger_km,
            )
            delivery_row = event_row(
                vehicle_id=vehicle_id,
                sequence=len(leg_rows),
                task=task,
                event_type="Delivery",
                movement_type="passenger_flight",
                from_node=task.origin,
                to_node=task.destination,
                departure_time=execution.passenger_departure,
                arrival_time=execution.passenger_arrival,
                waiting_time=0,
                service_time=SERVICE_TIME_MINUTES,
                passengers=task.passengers,
                range_at_departure=execution.departure_range_km,
                range_at_arrival=execution.range_after_passenger_km,
                charged_range=delivery_charge,
                flight_distance_km=task.distance_km,
                nodes=nodes,
                charge_remain=execution.range_after_delivery_service_km,
            )
            delivery_row["charging_time_min"] = execution.delivery_charge_time
            delivery_row["mandatory_charge_wait_min"] = execution.mandatory_delivery_charge_wait
            delivery_row["remaining_range_after_service_km"] = round(
                execution.range_after_delivery_service_km, 3
            )
            delivery_row["Charge_remain"] = round(
                execution.range_after_delivery_service_km, 3
            )
            leg_rows.append(delivery_row)

            task.status = "onboard"
            task.visit = True
            task.assigned_vehicle_id = vehicle_id
            task.actual_departure_time = execution.passenger_departure
            task.actual_arrival_time = execution.passenger_arrival
            task.completion_time = execution.passenger_arrival

            # 핵심: 같은 commit 구간의 다음 task가 이 운항 직후 상태에서 검증되도록
            # 위치/가용시각/배터리를 즉시 갱신한다.
            apply_execution_to_vehicle_state(state, task, execution)
            committed += 1

    return committed


def update_request_states(tasks: list[RequestTask], current_time: int) -> int:
    newly_overdue = 0
    for task in tasks:
        if (
            task.status == "onboard"
            and task.completion_time is not None
            and current_time >= task.completion_time
        ):
            task.status = "completed"
        elif (
            task.status in {"pending", "planned"}
            and current_time > task.max_available_departure
        ):
            task.status = "overdue"
            task.visit = False
            if not task.drop_penalty_applied:
                task.drop_penalty_applied = True
                newly_overdue += 1
    return newly_overdue


# ============================================================
# 결과 생성
# ============================================================
def request_status_dataframe(tasks: list[RequestTask]) -> pd.DataFrame:
    rows = []
    for task in tasks:
        rows.append(
            {
                "task_key": task.task_key,
                "request_id": task.request_id,
                "batch_id": task.batch_id,
                "n": task.origin,
                "m": task.destination,
                "cnt": task.passengers,
                "departure_time": task.desired_departure,
                "departure_time_hhmm": format_hhmm(task.desired_departure),
                "min_available_departure": task.min_available_departure,
                "max_available_departure": task.max_available_departure,
                "movement_time": task.ground_movement_time_min,
                "Joby_Flight_Time": task.joby_flight_time_min,
                "distance": task.distance_km,
                "planned_visit": task.status == "planned",
                "pickup_departure_executed": task.actual_departure_time is not None,
                "delivery_completed": task.status == "completed",
                "Visit": task.visit,
                "Status": task.status,
                "assigned_vehicle_id": task.assigned_vehicle_id,
                "actual_departure_time": task.actual_departure_time,
                "actual_departure_hhmm": format_hhmm(task.actual_departure_time),
                "actual_arrival_time": task.actual_arrival_time,
                "actual_arrival_hhmm": format_hhmm(task.actual_arrival_time),
                "request_drop_penalty": task.drop_penalty,
                "time_performance_penalty": task.time_performance_penalty,
                "drop_penalty_applied": task.drop_penalty_applied,
            }
        )
    return pd.DataFrame(rows).astype({"Visit": "boolean"})


def occupancy_dataframe(
    stop_rows: list[dict[str, Any]],
    nodes: dict[str, NodeInfo],
) -> pd.DataFrame:
    minute_rows: list[dict[str, Any]] = []
    for row in stop_rows:
        for minute in range(int(row["start_time_min"]), int(row["end_time_min"])):
            minute_rows.append(
                {
                    "time_min": minute,
                    "route_node_id": row["route_node_id"],
                    "vehicle_id": row["vehicle_id"],
                }
            )

    if not minute_rows:
        return pd.DataFrame(
            columns=[
                "time_min",
                "time_hhmm",
                "route_node_id",
                "physical_node_id",
                "physical_address",
                "vehicle_count",
                "vehicle_ids",
            ]
        )

    minute_df = pd.DataFrame(minute_rows)
    grouped = (
        minute_df.groupby(["time_min", "route_node_id"], as_index=False)
        .agg(
            vehicle_count=("vehicle_id", "nunique"),
            vehicle_ids=(
                "vehicle_id",
                lambda values: ",".join(map(str, sorted(set(values)))),
            ),
        )
    )
    grouped["time_hhmm"] = grouped["time_min"].map(format_hhmm)
    grouped["physical_node_id"] = grouped["route_node_id"].map(
        lambda value: nodes[value].physical_node_id
    )
    grouped["physical_address"] = grouped["route_node_id"].map(
        lambda value: nodes[value].address
    )
    return grouped[
        [
            "time_min",
            "time_hhmm",
            "route_node_id",
            "physical_node_id",
            "physical_address",
            "vehicle_count",
            "vehicle_ids",
        ]
    ]


def vehicle_summary_dataframe(
    route_df: pd.DataFrame,
    tasks: list[RequestTask],
    vehicle_states: list[VehicleState],
) -> pd.DataFrame:
    task_by_vehicle: dict[int, list[RequestTask]] = {}
    for task in tasks:
        if task.assigned_vehicle_id is not None:
            task_by_vehicle.setdefault(task.assigned_vehicle_id, []).append(task)

    flight_types = {"empty_repositioning", "passenger_flight", "return_to_depot"}
    rows: list[dict[str, Any]] = []

    for state in sorted(vehicle_states, key=lambda item: item.vehicle_id):
        vehicle_id = state.vehicle_id
        if route_df.empty:
            vehicle_rows = pd.DataFrame()
        else:
            vehicle_rows = route_df[route_df["vehicle_id"] == vehicle_id].copy()

        pickup_service_time = (
            int(
                vehicle_rows.loc[
                    vehicle_rows["event_type"] == "Pickup", "service_time_min"
                ].sum()
            )
            if not vehicle_rows.empty
            else 0
        )
        flights = (
            vehicle_rows[vehicle_rows["movement_type"].isin(flight_types)].copy()
            if not vehicle_rows.empty
            else pd.DataFrame()
        )

        if not flights.empty:
            flights = flights.sort_values(["departure_time_min", "arrival_time_min"])
            flight_start = int(flights["departure_time_min"].min())
            flight_end = int(flights["arrival_time_min"].max())
            total_flight_time = int(flights["flight_time_min"].sum())
            total_flight_distance = float(flights["flight_distance_km"].sum())

            passenger_flights = flights[flights["movement_type"] == "passenger_flight"]
            empty_flights = flights[flights["movement_type"] == "empty_repositioning"]

            passenger_flight_time = int(passenger_flights["flight_time_min"].sum())
            empty_flight_time = int(empty_flights["flight_time_min"].sum())
            passenger_distance = float(passenger_flights["flight_distance_km"].sum())
            empty_distance = float(empty_flights["flight_distance_km"].sum())
            cumulative_passengers = int(passenger_flights["passenger_load"].sum())

            passenger_minutes = float(
                (flights["passenger_load"] * flights["flight_time_min"]).sum()
            )
            average_onboard = (
                passenger_minutes / total_flight_time if total_flight_time > 0 else 0.0
            )
            average_onboard = min(float(VEHICLE_CAPACITY), max(0.0, average_onboard))

            final_flight = flights.iloc[-1]
            final_remaining_range = float(final_flight["remaining_range_at_arrival_km"])
            fuel_shortage_count = int(
                (flights["mandatory_charge_wait_min"].fillna(0) > 0).sum()
            )
            empty_count = int(len(empty_flights))
            passenger_count = int(len(passenger_flights))
        else:
            flight_start = None
            flight_end = None
            total_flight_time = 0
            total_flight_distance = 0.0
            passenger_flight_time = 0
            empty_flight_time = 0
            passenger_distance = 0.0
            empty_distance = 0.0
            cumulative_passengers = 0
            average_onboard = 0.0
            final_remaining_range = float(state.remaining_range_km)
            fuel_shortage_count = 0
            empty_count = 0
            passenger_count = 0

        assigned_tasks = task_by_vehicle.get(vehicle_id, [])
        assigned_request_penalty_score = sum(task.drop_penalty for task in assigned_tasks)
        assigned_time_performance_penalty = sum(
            task.time_performance_penalty for task in assigned_tasks
        )

        rows.append(
            {
                "vehicle_id": vehicle_id,
                "boarding_time_min": pickup_service_time,
                "pickup_service_time_min": pickup_service_time,
                "passenger_carrying_time_min": passenger_flight_time,
                "empty_flight_time_min": empty_flight_time,
                "flight_time_min": total_flight_time,
                "total_flight_time_min": total_flight_time,
                "total_flight_distance_km": round(total_flight_distance, 3),
                "passenger_flight_distance_km": round(passenger_distance, 3),
                "empty_flight_distance_km": round(empty_distance, 3),
                "final_remaining_range_km": round(final_remaining_range, 3),
                "flight_start_time_min": flight_start,
                "flight_start_time_hhmm": format_hhmm(flight_start),
                "flight_end_time_min": flight_end,
                "flight_end_time_hhmm": format_hhmm(flight_end),
                "flight_operation_span_min": (
                    0
                    if flight_start is None or flight_end is None
                    else flight_end - flight_start
                ),
                "fuel_shortage_count": fuel_shortage_count,
                "empty_movement_count": empty_count,
                "passenger_movement_count": passenger_count,
                "cumulative_passengers": cumulative_passengers,
                "average_onboard_passengers": round(average_onboard, 3),
                "penalty": int(assigned_request_penalty_score),
                "assigned_request_penalty_score": int(assigned_request_penalty_score),
                "assigned_time_performance_penalty": int(
                    assigned_time_performance_penalty
                ),
            }
        )

    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")


# ============================================================
# 메인 Rolling Horizon
# ============================================================
def main() -> None:
    start_clock = time.perf_counter()
    ensure_cuopt_available()

    if TIME_LIMIT_SECONDS <= 0:
        raise ValueError("TIME_LIMIT_SECONDS는 0보다 커야 합니다.")
    if ROLLING_HORIZON_MINUTES <= 0 or REOPTIMIZATION_INTERVAL_MINUTES <= 0:
        raise ValueError("Rolling Horizon/재최적화 간격은 1분 이상이어야 합니다.")

    distance_matrix_df = load_matrix(DISTANCE_MATRIX_PATH, integer=False)
    time_matrix_df = load_matrix(TIME_MATRIX_PATH, integer=True)
    nodes = load_nodes(NODE_REFERENCE_PATH)

    if set(distance_matrix_df.index) != set(time_matrix_df.index):
        raise ValueError("거리행렬과 시간행렬의 노드가 일치하지 않습니다.")
    if not set(distance_matrix_df.index).issubset(nodes):
        raise ValueError("행렬 노드 중 노드 참조표에 없는 노드가 있습니다.")

    matrices = build_cuopt_matrix_store(distance_matrix_df, time_matrix_df)

    depot_ids = [normalize_id(value) for value in DEPOT_ROUTE_NODE_IDS]
    missing_depots = [value for value in depot_ids if value not in nodes]
    if missing_depots:
        raise ValueError(f"Depot 노드가 참조표에 없습니다: {missing_depots}")

    tasks = load_tasks(REQUEST_PATH, matrices, nodes)
    if not tasks:
        raise ValueError("유효한 요청 task가 없습니다.")

    task_lookup = {task.task_key: task for task in tasks}
    vehicle_states = [
        VehicleState(
            vehicle_id=vehicle_id,
            route_node_id=depot_ids[vehicle_id % len(depot_ids)],
            node_arrival_time=0,
            available_time=0,
            remaining_range_km=INITIAL_REMAINING_RANGE_KM,
        )
        for vehicle_id in range(NUM_VEHICLES)
    ]

    plan_history: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    route_leg_rows: list[dict[str, Any]] = []
    stop_rows: list[dict[str, Any]] = []

    maximum_departure = max(task.max_available_departure for task in tasks)
    simulation_end = (
        maximum_departure
        + matrices.maximum_flight_time
        + SERVICE_TIME_MINUTES
        + REQUEST_TAKE_OFF_TIME_MINUTES
        + ARRIVAL_BUFFER_MINUTES
        + 1
    )

    horizon_starts = range(
        0,
        simulation_end + 1,
        REOPTIMIZATION_INTERVAL_MINUTES,
    )
    total_horizons = len(horizon_starts)
    print(
        f"cuOpt Rolling Horizon 시작: 요청 {len(tasks):,}개, "
        f"차량 {NUM_VEHICLES}대, 총 {total_horizons}개 구간",
        flush=True,
    )
    print(
        f"cuOpt 설정: time_limit={TIME_LIMIT_SECONDS:.1f}초/Solve | "
        f"가용 fleet={NUM_VEHICLES}대 | "
        f"min_vehicles={'미강제' if CUOPT_MIN_VEHICLES is None else CUOPT_MIN_VEHICLES}",
        flush=True,
    )

    for horizon_index, horizon_start in enumerate(horizon_starts, start=1):
        horizon_clock = time.perf_counter()
        horizon_end = horizon_start + ROLLING_HORIZON_MINUTES
        commit_end = horizon_start + REOPTIMIZATION_INTERVAL_MINUTES
        update_request_states(tasks, horizon_start)

        active_tasks = [
            task
            for task in tasks
            if task.status in {"pending", "planned"}
            and task.min_available_departure <= horizon_end
            and task.max_available_departure >= horizon_start
        ]

        raw_cuopt_objective: float | None = None
        equivalent_objective: int | None = None
        route_cost_m = 0
        horizon_drop_penalty = 0
        horizon_time_performance_penalty = 0
        planned_count = 0
        committed_count = 0
        solver_seconds = 0.0
        repair_resolves = 0
        cuopt_status: int | None = None
        cuopt_vehicle_count: int | None = None

        print(
            f"[{horizon_index:02d}/{total_horizons:02d}] "
            f"구간 {format_hhmm(horizon_start)}~{format_hhmm(horizon_end)} | "
            f"활성 요청 {len(active_tasks):,}개 | 최적화 중...",
            flush=True,
        )

        if active_tasks:
            (
                model,
                solution,
                route_plan,
                task_plan,
                cuopt_route_df,
                solver_seconds,
                repair_resolves,
            ) = solve_horizon_with_battery_repair(
                active_tasks,
                vehicle_states,
                matrices,
                horizon_start,
                horizon_end,
                commit_end,
            )

            if solution is not None:
                try:
                    cuopt_status = int(solution.get_status())
                except Exception:
                    cuopt_status = None
                try:
                    raw_cuopt_objective = float(solution.get_total_objective())
                except Exception:
                    raw_cuopt_objective = None
                try:
                    cuopt_vehicle_count = int(solution.get_vehicle_count())
                except Exception:
                    cuopt_vehicle_count = None

            route_cost_m = compute_planned_route_cost_m(cuopt_route_df, matrices)
            (
                equivalent_objective,
                horizon_drop_penalty,
                horizon_time_performance_penalty,
            ) = compute_equivalent_horizon_objective(
                active_tasks,
                task_plan,
                route_cost_m,
            )

            for task in active_tasks:
                plan = task_plan[task.task_key]
                planned_count += int(plan["planned_visit"])
                if plan["planned_visit"]:
                    task.status = "planned"
                plan_history.append(
                    {
                        "horizon_start": horizon_start,
                        "horizon_end": horizon_end,
                        "task_key": task.task_key,
                        "request_id": task.request_id,
                        "batch_id": task.batch_id,
                        "planned_visit": plan["planned_visit"],
                        "planned_vehicle_id": plan["vehicle_id"],
                        "planned_departure": plan["planned_departure"],
                        "planned_departure_hhmm": format_hhmm(plan["planned_departure"]),
                        "planned_arrival": plan["planned_arrival"],
                        "planned_arrival_hhmm": format_hhmm(plan["planned_arrival"]),
                    }
                )

            committed_count = commit_all_committable_tasks(
                route_plan,
                task_lookup,
                vehicle_states,
                matrices,
                nodes,
                horizon_start,
                commit_end,
                route_leg_rows,
                stop_rows,
            )

            # 미래 계획은 다음 Horizon에서 다시 계산한다.
            for task in active_tasks:
                if task.status == "planned":
                    task.status = "pending"

        update_request_states(tasks, commit_end)

        summary_rows.append(
            {
                "horizon_start": horizon_start,
                "horizon_end": horizon_end,
                "new_request_count": sum(
                    horizon_start <= task.desired_departure < commit_end for task in tasks
                ),
                "active_request_count": len(active_tasks),
                "pending_request_count": sum(task.status == "pending" for task in tasks),
                "planned_request_count": planned_count,
                "committed_request_count": committed_count,
                "onboard_request_count": sum(task.status == "onboard" for task in tasks),
                "completed_request_count": sum(task.status == "completed" for task in tasks),
                "overdue_request_count": sum(task.status == "overdue" for task in tasks),
                "drop_penalty_total": sum(
                    task.drop_penalty for task in tasks if task.drop_penalty_applied
                ),
                "horizon_planned_drop_penalty": horizon_drop_penalty,
                "time_performance_penalty_total": horizon_time_performance_penalty,
                "route_cost": route_cost_m,
                "objective_value": equivalent_objective,
                "cuopt_raw_objective": raw_cuopt_objective,
                "cuopt_status": cuopt_status,
                "cuopt_vehicle_count": cuopt_vehicle_count,
                "battery_repair_resolves": repair_resolves,
                "solver_runtime_seconds": round(solver_seconds, 6),
            }
        )

        print(
            f"[{horizon_index:02d}/{total_horizons:02d}] 완료 | "
            f"계획 {planned_count:,}개 | 확정 {committed_count:,}개 | "
            f"누적 완료 {sum(task.status == 'completed' for task in tasks):,}개 | "
            f"Overdue {sum(task.status == 'overdue' for task in tasks):,}개 | "
            f"solver {solver_seconds:.2f}초 | "
            f"구간 전체 {time.perf_counter() - horizon_clock:.2f}초",
            flush=True,
        )

    update_request_states(tasks, simulation_end + 1)

    final_stop_end = max(
        simulation_end,
        max((state.available_time for state in vehicle_states), default=simulation_end),
    )
    for state in vehicle_states:
        add_stop_interval(
            stop_rows,
            state.vehicle_id,
            state.route_node_id,
            state.node_arrival_time,
            final_stop_end,
        )

    request_df = request_status_dataframe(tasks)
    route_df = pd.DataFrame(route_leg_rows)
    if not route_df.empty:
        route_df = route_df.sort_values(
            ["vehicle_id", "departure_time_min", "arrival_time_min", "event_type"]
        ).reset_index(drop=True)
        route_df["route_sequence"] = route_df.groupby("vehicle_id").cumcount()

    occupancy_df = occupancy_dataframe(stop_rows, nodes)
    plan_df = pd.DataFrame(plan_history)
    summary_df = pd.DataFrame(summary_rows)
    vehicle_summary_df = vehicle_summary_dataframe(route_df, tasks, vehicle_states)

    save_csv(request_df, "rolling_horizon_request_status.csv")
    save_csv(route_df, "vehicle_route_legs.csv")
    save_csv(occupancy_df, "node_vehicle_occupancy_by_minute.csv")
    save_csv(plan_df, "rolling_horizon_plan_history.csv")
    save_csv(summary_df, "rolling_horizon_summary.csv")
    save_csv(vehicle_summary_df, "vehicle_summary.csv")

    print("\n========== cuOpt 실행 결과 ==========")
    print(f"GPU: {matrices.device_name}")
    print(f"cuOpt Routing: GPU solver")
    print(f"요청 batch 수: {len(tasks):,}")
    print(f"완료: {sum(task.status == 'completed' for task in tasks):,}")
    print(f"운항 중: {sum(task.status == 'onboard' for task in tasks):,}")
    print(f"Overdue: {sum(task.status == 'overdue' for task in tasks):,}")
    print(f"Vehicle summary: {len(vehicle_summary_df):,}대")
    print(f"실행시간: {time.perf_counter() - start_clock:.2f}초")
    print(f"결과 폴더: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\n사용자가 Ctrl+C로 실행을 중단했습니다. "
            "최종 저장 전에 중단된 경우 결과 CSV는 갱신되지 않습니다.",
            flush=True,
        )
        raise SystemExit(130)
