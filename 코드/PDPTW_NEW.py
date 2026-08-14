# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except ImportError:  # 입력 자료만 점검할 때도 파일을 import할 수 있도록 함
    pywrapcp = None
    routing_enums_pb2 = None


# ============================================================
# 사용자 설정: 아래 경로와 값만 수정하면 됩니다.
# ============================================================
INPUT_DIR = Path(
    r"C:\Users\choih\OneDrive\Desktop\연구\PDPTW\자료\기초자료"
)
REQUEST_PATH = INPUT_DIR / "d5000_s20.csv"
DISTANCE_MATRIX_PATH = INPUT_DIR / "Distance_matrix_1_9.csv"
TIME_MATRIX_PATH = INPUT_DIR / "Time_matrix_1_9.csv"
NODE_REFERENCE_PATH = INPUT_DIR / "vp_reference_remove_동탄.csv"
OUTPUT_DIR = Path(
    r"C:\Users\c\Desktop\대학생활\학연생\새로운_경로방식\자료\심화\교수님_자료\결과_보고서용"
)

BASE_TIME = "05:40"
NUM_VEHICLES = 251
VEHICLE_CAPACITY = 3

# 수요·행렬의 노드 ID를 입력합니다. 업로드 자료에서는 vp_id(1~9)와 같습니다.
DEPOT_ROUTE_NODE_IDS = [1 ,2, 3, 4, 5, 6, 7, 8, 9]

# 새로 구간을 생성하는 구간단위(ex. 20분 간격 -> [0,30], [30,50], [50,70]....)
ROLLING_HORIZON_MINUTES = 30
# 측정할 간격
REOPTIMIZATION_INTERVAL_MINUTES = 20
# 각 재최적화 단계에서 솔버가 계산에 사용할 최대 시간(초)입니다.
TIME_LIMIT_SECONDS = 20

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

LOG_SEARCH = False


# 여러 자료에서 사용될 수 있는 열 이름 후보입니다.
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
class HorizonModel:
    manager: Any
    routing: Any
    time_dimension: Any
    range_dimension: Any
    task_by_model_node: dict[int, RequestTask]
    model_node_by_task_key: dict[str, int]
    start_model_nodes: dict[int, int]
    end_model_nodes: set[int]
    start_vehicle_by_model_node: dict[int, int]


# ============================================================
# 입력 자료 처리
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


def compute_drop_penalty(passengers: int, distance_km: float, joby_time: int, ground_time: int) -> int:
    # 사용자의 정의: Joby가 지상교통보다 빠를수록 시간항이 음수가 되어 penalty가 작아짐.
    value = (
        DROP_PENALTY_BASE
        + DEMAND_PENALTY_WEIGHT * passengers
        + DISTANCE_PENALTY_WEIGHT * distance_km
        + TIME_PENALTY_WEIGHT * (joby_time - ground_time)
    )
    return max(MIN_REQUEST_DROP_PENALTY, int(round(value)))


def load_tasks(
    request_path: Path,
    distance_matrix: pd.DataFrame,
    time_matrix: pd.DataFrame,
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
        if origin not in distance_matrix.index or destination not in distance_matrix.columns:
            raise KeyError(f"수요 row={row_index}의 노드가 거리행렬에 없습니다: {origin}->{destination}")

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

    grouped = (
        pd.DataFrame(standardized)
        .groupby(["origin", "destination", "desired_departure", "max_wait", "ground_time"], as_index=False)
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
        distance = float(distance_matrix.loc[origin, destination])
        joby_time = int(time_matrix.loc[origin, destination])
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
                    drop_penalty=compute_drop_penalty(
                        batch_passengers,
                        distance,
                        joby_time,
                        int(row["ground_time"]),
                    ),
                )
            )
            remaining -= batch_passengers
            batch_id += 1
    return tasks


# ============================================================
# Rolling Horizon용 OR-Tools 모델
# 각 batch를 하나의 원자적 OD 작업으로 모델링합니다.
# 이전 작업의 도착지 → 다음 작업의 출발지는 공차 이동입니다.
# ============================================================
def build_horizon_model(
    active_tasks: list[RequestTask],
    vehicle_states: list[VehicleState],
    distance_matrix: pd.DataFrame,
    time_matrix: pd.DataFrame,
    horizon_start: int,
    horizon_end: int,
) -> HorizonModel:
    if pywrapcp is None:
        raise ImportError("ortools가 설치되어 있지 않습니다. 'pip install ortools' 후 실행해 주세요.")

    vehicle_count = len(vehicle_states)
    start_model_nodes = {vehicle.vehicle_id: vehicle.vehicle_id for vehicle in vehicle_states}
    end_start = vehicle_count
    end_model_nodes = {end_start + vehicle.vehicle_id for vehicle in vehicle_states}
    first_task_node = vehicle_count * 2

    task_by_model_node = {
        first_task_node + index: task for index, task in enumerate(active_tasks)
    }
    model_node_by_task_key = {
        task.task_key: model_node for model_node, task in task_by_model_node.items()
    }
    start_vehicle_by_model_node = {
        model_node: vehicle_id for vehicle_id, model_node in start_model_nodes.items()
    }
    vehicle_by_id = {vehicle.vehicle_id: vehicle for vehicle in vehicle_states}

    starts = [start_model_nodes[vehicle.vehicle_id] for vehicle in vehicle_states]
    ends = [end_start + vehicle.vehicle_id for vehicle in vehicle_states]
    manager = pywrapcp.RoutingIndexManager(
        vehicle_count * 2 + len(active_tasks), vehicle_count, starts, ends
    )
    routing = pywrapcp.RoutingModel(manager)

    def from_location(model_node: int) -> str | None:
        if model_node in start_vehicle_by_model_node:
            return vehicle_by_id[start_vehicle_by_model_node[model_node]].route_node_id
        task = task_by_model_node.get(model_node)
        return None if task is None else task.destination

    def to_location(model_node: int) -> str | None:
        task = task_by_model_node.get(model_node)
        return None if task is None else task.origin

    # Movement, pickup, delivery 관련된(OD) 거리 정보들을 담고 있음
    def passenger_distance(model_node: int) -> float:
        task = task_by_model_node.get(model_node)
        return 0.0 if task is None else task.distance_km

    # Movement, pickup, delivery 관련된(OD) 시간 정보들을 담고 있음
    def passenger_time(model_node: int) -> int:
        task = task_by_model_node.get(model_node)
        return 0 if task is None else task.joby_flight_time_min

    # 이전 위치에서 다음 Request의 Pickup 위치까지 이동하는
    # 공차 이동(Empty Repositioning) 거리
    def reposition_distance(from_node: int, to_node: int) -> float:
        from_route = from_location(from_node)
        to_route = to_location(to_node)
        if from_route is None or to_route is None:
            return 0.0
        return float(distance_matrix.loc[from_route, to_route])

    # 이전 위치에서 다음 Request의 Pickup 위치까지 이동하는
    # 공차 이동(Empty Repositioning) 시간
    def reposition_time(from_node: int, to_node: int) -> int:
        from_route = from_location(from_node)
        to_route = to_location(to_node)
        if from_route is None or to_route is None:
            return 0
        return int(time_matrix.loc[from_route, to_route])

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(round(1000 * (passenger_distance(from_node) + reposition_distance(from_node, to_node))))

    # 각 상황별 departure 조건을 생성
    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if from_node in start_vehicle_by_model_node and to_node in end_model_nodes:
            return 0
        # depot에서 시작한 경우 departure 시점
        if from_node in start_vehicle_by_model_node:
            return reposition_time(from_node, to_node) + SERVICE_TIME_MINUTES + REQUEST_TAKE_OFF_TIME_MINUTES
        # depot 외의 다른 노드에서 시작한 경우의 departure 시점
        if to_node in end_model_nodes:
            return passenger_time(from_node) + SERVICE_TIME_MINUTES
        return (
            passenger_time(from_node)
            + SERVICE_TIME_MINUTES
            + reposition_time(from_node, to_node)
            + SERVICE_TIME_MINUTES
            + REQUEST_TAKE_OFF_TIME_MINUTES
        )

    def range_callback(from_index: int, to_index: int) -> int:
        # Range Cumul은 실제 남은 운항거리(m)이며, 비행거리는 음수 transit으로 반영합니다.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        used_km = passenger_distance(from_node) + reposition_distance(from_node, to_node)
        return -int(round(used_km * 1000))

    distance_callback_index = routing.RegisterTransitCallback(distance_callback)
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    range_callback_index = routing.RegisterTransitCallback(range_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

    maximum_flight_time = int(time_matrix.to_numpy().max())
    model_end = horizon_end + maximum_flight_time + SERVICE_TIME_MINUTES + ARRIVAL_BUFFER_MINUTES
    routing.AddDimension(time_callback_index, model_end, model_end, False, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")

    for vehicle in vehicle_states:
        start_index = routing.Start(vehicle.vehicle_id)
        start_time = max(horizon_start, int(vehicle.available_time))
        time_dimension.CumulVar(start_index).SetValue(start_time)
        time_dimension.CumulVar(routing.End(vehicle.vehicle_id)).SetRange(start_time, model_end)
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(vehicle.vehicle_id)))

    solver = routing.solver()
    for model_node, task in task_by_model_node.items():
        index = manager.NodeToIndex(model_node)
        lower = max(horizon_start, task.min_available_departure)
        upper = min(horizon_end, task.max_available_departure)
        time_dimension.CumulVar(index).SetRange(lower, upper)
        routing.AddDisjunction([index], task.drop_penalty)

    max_range_m = int(round(MAX_REMAINING_RANGE_KM * 1000))
    min_range_m = int(round(MIN_REMAINING_RANGE_KM * 1000))
    charge_rate_m = int(round(CHARGING_RATE_KM_PER_MIN * 1000))
    routing.AddDimension(range_callback_index, max_range_m, max_range_m, False, "RemainingRange")
    range_dimension = routing.GetDimensionOrDie("RemainingRange")

    for vehicle in vehicle_states:
        start_index = routing.Start(vehicle.vehicle_id)
        start_range = int(round(vehicle.remaining_range_km * 1000))
        range_dimension.CumulVar(start_index).SetValue(start_range)
        range_dimension.CumulVar(routing.End(vehicle.vehicle_id)).SetRange(0, max_range_m)
        range_dimension.SlackVar(start_index).SetRange(0, max_range_m)
        solver.Add(
            range_dimension.SlackVar(start_index)
            <= charge_rate_m * (time_dimension.SlackVar(start_index) + SERVICE_TIME_MINUTES)
        )
        solver.Add(range_dimension.CumulVar(start_index) + range_dimension.SlackVar(start_index) <= max_range_m)

    for model_node, task in task_by_model_node.items():
        index = manager.NodeToIndex(model_node)
        direct_distance_m = int(round(task.distance_km * 1000))
        # 승객 운항 종료 시 최소 15km를 남길 수 있는 출발 Range를 요구합니다.
        range_dimension.CumulVar(index).SetRange(direct_distance_m + min_range_m, max_range_m)
        range_dimension.SlackVar(index).SetRange(0, max_range_m)
        # Delivery service + 다음 Pickup service 및 대기시간 동안 충전할 수 있습니다.
        solver.Add(
            range_dimension.SlackVar(index)
            <= charge_rate_m * (
                time_dimension.SlackVar(index) + SERVICE_TIME_MINUTES * 2
            )
        )
        # 승객 운항 도착 후 충전량까지 포함해 160km를 넘지 않도록 제한합니다.
        solver.Add(
            range_dimension.CumulVar(index)
            - direct_distance_m
            + range_dimension.SlackVar(index)
            <= max_range_m
        )

    return HorizonModel(
        manager=manager,
        routing=routing,
        time_dimension=time_dimension,
        range_dimension=range_dimension,
        task_by_model_node=task_by_model_node,
        model_node_by_task_key=model_node_by_task_key,
        start_model_nodes=start_model_nodes,
        end_model_nodes=end_model_nodes,
        start_vehicle_by_model_node=start_vehicle_by_model_node,
    )


def solve_horizon(model: HorizonModel):
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parameters.time_limit.FromSeconds(TIME_LIMIT_SECONDS)
    parameters.log_search = LOG_SEARCH
    return model.routing.SolveWithParameters(parameters)


def extract_plan(model: HorizonModel, solution: Any, vehicle_count: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    route_plan: list[dict[str, Any]] = []
    task_plan: dict[str, dict[str, Any]] = {
        task.task_key: {"planned_visit": False, "vehicle_id": None, "planned_departure": None}
        for task in model.task_by_model_node.values()
    }
    if solution is None:
        return route_plan, task_plan

    for vehicle_id in range(vehicle_count):
        index = model.routing.Start(vehicle_id)
        sequence = 0
        while not model.routing.IsEnd(index):
            next_index = solution.Value(model.routing.NextVar(index))
            next_model_node = model.manager.IndexToNode(next_index)
            task = model.task_by_model_node.get(next_model_node)
            if task is not None:
                departure = int(solution.Value(model.time_dimension.CumulVar(next_index)))
                departure_range = solution.Value(model.range_dimension.CumulVar(next_index)) / 1000.0
                row = {
                    "vehicle_id": vehicle_id,
                    "sequence": sequence,
                    "task_key": task.task_key,
                    "planned_departure": departure,
                    "remaining_range_at_passenger_departure": departure_range,
                }
                route_plan.append(row)
                task_plan[task.task_key] = {
                    "planned_visit": True,
                    "vehicle_id": vehicle_id,
                    "planned_departure": departure,
                }
                sequence += 1
            index = next_index
    return route_plan, task_plan


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


def commit_first_tasks(
    route_plan: list[dict[str, Any]],
    task_lookup: dict[str, RequestTask],
    vehicle_states: list[VehicleState],
    distance_matrix: pd.DataFrame,
    time_matrix: pd.DataFrame,
    nodes: dict[str, NodeInfo],
    horizon_start: int,
    commit_end: int,
    leg_rows: list[dict[str, Any]],
    stop_rows: list[dict[str, Any]],
) -> int:
    state_by_vehicle = {state.vehicle_id: state for state in vehicle_states}
    first_by_vehicle: dict[int, dict[str, Any]] = {}
    for row in sorted(route_plan, key=lambda item: (item["vehicle_id"], item["sequence"])):
        if row["planned_departure"] < commit_end and row["planned_departure"] >= horizon_start:
            first_by_vehicle.setdefault(row["vehicle_id"], row)

    committed = 0
    for vehicle_id, plan in first_by_vehicle.items():
        task = task_lookup[plan["task_key"]]
        if task.status not in {"pending", "planned"}:
            continue

        state = state_by_vehicle[vehicle_id]
        passenger_departure = int(plan["planned_departure"])
        empty_distance = float(distance_matrix.loc[state.route_node_id, task.origin])
        empty_time = int(time_matrix.loc[state.route_node_id, task.origin])

        total_wait = max(
            0,
            passenger_departure
            - int(state.available_time)
            - empty_time
            - SERVICE_TIME_MINUTES
            - REQUEST_TAKE_OFF_TIME_MINUTES,
        )
        # 공차 이동 도착 시에도 15km 이상 남도록 필요한 대기의 일부를 현재 노드에 배치합니다.
        required_precharge = max(
            0.0,
            MIN_REMAINING_RANGE_KM + empty_distance - state.remaining_range_km,
        )
        pre_reposition_wait = min(
            total_wait,
            int(math.ceil(required_precharge / CHARGING_RATE_KM_PER_MIN)),
        )
        pickup_wait = total_wait - pre_reposition_wait
        empty_departure = int(state.available_time) + pre_reposition_wait
        empty_arrival = empty_departure + empty_time
        pickup_start = empty_arrival

        # 현재 노드의 Service/Waiting/Charging 체류를 다음 이동 직전까지 기록합니다.
        add_stop_interval(
            stop_rows,
            vehicle_id,
            state.route_node_id,
            state.node_arrival_time,
            empty_departure,
        )

        range_before_empty = min(
            MAX_REMAINING_RANGE_KM,
            state.remaining_range_km
            + CHARGING_RATE_KM_PER_MIN * pre_reposition_wait,
        )
        range_after_empty = range_before_empty - empty_distance
        pickup_charge_time = pickup_wait + SERVICE_TIME_MINUTES
        mandatory_pickup_charge_wait = 0
        low_range_after_empty = range_after_empty <= MIN_REMAINING_RANGE_KM
        if low_range_after_empty:
            required_total = math.ceil(
                max(0.0, LOW_RANGE_CHARGE_TARGET_KM - range_after_empty)
                / CHARGING_RATE_KM_PER_MIN
            )
            mandatory_pickup_charge_wait = max(0, required_total - pickup_charge_time)
            pickup_wait += mandatory_pickup_charge_wait
            passenger_departure += mandatory_pickup_charge_wait
            pickup_charge_time = max(pickup_charge_time, required_total)
            computed_departure_range = min(
                MAX_REMAINING_RANGE_KM,
                LOW_RANGE_CHARGE_TARGET_KM,
            )
        else:
            pickup_charge = CHARGING_RATE_KM_PER_MIN * pickup_charge_time
            computed_departure_range = min(
                MAX_REMAINING_RANGE_KM,
                range_after_empty + pickup_charge,
            )

        if computed_departure_range < MIN_REMAINING_RANGE_KM - 1e-6:
            raise RuntimeError(
                f"공차 이동 후 충전해도 최소 잔여거리 위반: vehicle={vehicle_id}, "
                f"request={task.task_key}, remain={computed_departure_range:.3f}km"
            )

        model_departure_range = float(plan["remaining_range_at_passenger_departure"])
        departure_range = (
            computed_departure_range
            if low_range_after_empty
            else min(computed_departure_range, model_departure_range)
        )
        if departure_range < task.distance_km + MIN_REMAINING_RANGE_KM - 1e-6:
            raise RuntimeError(
                f"승객 운항 후 최소 잔여거리 위반: vehicle={vehicle_id}, "
                f"request={task.task_key}, departure_range={departure_range:.3f}km"
            )

        if empty_distance > 0:
            movement_row = event_row(
                vehicle_id=vehicle_id,
                sequence=len(leg_rows),
                task=task,
                event_type="Movement",
                movement_type="empty_repositioning",
                from_node=state.route_node_id,
                to_node=task.origin,
                departure_time=empty_departure,
                arrival_time=empty_arrival,
                waiting_time=0,
                service_time=0,
                passengers=0,
                range_at_departure=range_before_empty,
                range_at_arrival=range_after_empty,
                charged_range=max(0.0, range_before_empty - state.remaining_range_km),
                flight_distance_km=empty_distance,
                nodes=nodes,
            )
            movement_row["charging_time_min"] = pre_reposition_wait
            movement_row["mandatory_charge_wait_min"] = pre_reposition_wait
            leg_rows.append(movement_row)

        add_stop_interval(stop_rows, vehicle_id, task.origin, pickup_start, passenger_departure)
        pickup_row = event_row(
            vehicle_id=vehicle_id,
            sequence=len(leg_rows),
            task=task,
            event_type="Pickup",
            movement_type="pickup_service",
            from_node=task.origin,
            to_node=task.origin,
            departure_time=passenger_departure,
            arrival_time=pickup_start,
            waiting_time=pickup_wait,
            service_time=SERVICE_TIME_MINUTES,
            passengers=task.passengers,
            range_at_departure=departure_range,
            range_at_arrival=range_after_empty,
            charged_range=max(0.0, departure_range - range_after_empty),
            flight_distance_km=0.0,
            nodes=nodes,
            charge_remain=departure_range,
        )
        pickup_row["charging_time_min"] = pickup_charge_time
        pickup_row["mandatory_charge_wait_min"] = mandatory_pickup_charge_wait
        leg_rows.append(pickup_row)

        passenger_arrival = passenger_departure + task.joby_flight_time_min
        range_after_passenger = departure_range - task.distance_km
        delivery_charge_time = SERVICE_TIME_MINUTES
        low_range_violation = range_after_passenger <= MIN_REMAINING_RANGE_KM
        if low_range_violation:
            required_total = math.ceil(
                max(0.0, LOW_RANGE_CHARGE_TARGET_KM - range_after_passenger)
                / CHARGING_RATE_KM_PER_MIN
            )
            delivery_charge_time = max(SERVICE_TIME_MINUTES, required_total)

        if low_range_violation:
            # 충전시간은 분 단위로 올림하지만, 잔여 주행거리는 목표값인 100km를
            # 초과하지 않도록 실제 충전량을 목표값에서 정확히 제한합니다.
            range_after_delivery_service = min(
                MAX_REMAINING_RANGE_KM,
                LOW_RANGE_CHARGE_TARGET_KM,
            )
            delivery_charge = max(
                0.0,
                range_after_delivery_service - range_after_passenger,
            )
        else:
            delivery_charge = CHARGING_RATE_KM_PER_MIN * delivery_charge_time
            range_after_delivery_service = min(
                MAX_REMAINING_RANGE_KM,
                range_after_passenger + delivery_charge,
            )
        mandatory_charge_wait = max(0, delivery_charge_time - SERVICE_TIME_MINUTES)
        delivery_row = event_row(
            vehicle_id=vehicle_id,
            sequence=len(leg_rows),
            task=task,
            event_type="Delivery",
            movement_type="passenger_flight",
            from_node=task.origin,
            to_node=task.destination,
            departure_time=passenger_departure,
            arrival_time=passenger_arrival,
            waiting_time=0,
            service_time=SERVICE_TIME_MINUTES,
            passengers=task.passengers,
            range_at_departure=departure_range,
            range_at_arrival=range_after_passenger,
            charged_range=delivery_charge,
            flight_distance_km=task.distance_km,
            nodes=nodes,
            charge_remain=range_after_delivery_service,
        )
        delivery_row["charging_time_min"] = delivery_charge_time
        delivery_row["mandatory_charge_wait_min"] = mandatory_charge_wait
        delivery_row["remaining_range_after_service_km"] = round(range_after_delivery_service, 3)
        delivery_row["Charge_remain"] = round(range_after_delivery_service, 3)
        leg_rows.append(delivery_row)

        completion_time = passenger_arrival
        available_time = passenger_arrival + delivery_charge_time

        task.status = "onboard"
        task.visit = True
        task.assigned_vehicle_id = vehicle_id
        task.actual_departure_time = passenger_departure
        task.actual_arrival_time = passenger_arrival
        task.completion_time = completion_time

        state.route_node_id = task.destination
        state.node_arrival_time = passenger_arrival
        state.available_time = available_time
        state.remaining_range_km = range_after_delivery_service
        committed += 1

    return committed


def update_request_states(tasks: list[RequestTask], current_time: int) -> int:
    newly_overdue = 0
    for task in tasks:
        if task.status == "onboard" and task.completion_time is not None and current_time >= task.completion_time:
            task.status = "completed"
        elif task.status in {"pending", "planned"} and current_time > task.max_available_departure:
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
    return pd.DataFrame(
        [
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
                "Visit": task.visit,
                "Status": task.status,
                "assigned_vehicle_id": task.assigned_vehicle_id,
                "actual_departure_time": task.actual_departure_time,
                "actual_departure_hhmm": format_hhmm(task.actual_departure_time),
                "actual_arrival_time": task.actual_arrival_time,
                "actual_arrival_hhmm": format_hhmm(task.actual_arrival_time),
                "request_drop_penalty": task.drop_penalty,
                "drop_penalty_applied": task.drop_penalty_applied,
            }
            for task in tasks
        ]
    ).astype({"Visit": "boolean"})


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
            columns=["time_min", "time_hhmm", "route_node_id", "physical_node_id", "physical_address", "vehicle_count", "vehicle_ids"]
        )

    minute_df = pd.DataFrame(minute_rows)
    grouped = (
        minute_df.groupby(["time_min", "route_node_id"], as_index=False)
        .agg(
            vehicle_count=("vehicle_id", "nunique"),
            vehicle_ids=("vehicle_id", lambda values: ",".join(map(str, sorted(set(values))))),
        )
    )
    grouped["time_hhmm"] = grouped["time_min"].map(format_hhmm)
    grouped["physical_node_id"] = grouped["route_node_id"].map(lambda value: nodes[value].physical_node_id)
    grouped["physical_address"] = grouped["route_node_id"].map(lambda value: nodes[value].address)
    return grouped[
        ["time_min", "time_hhmm", "route_node_id", "physical_node_id", "physical_address", "vehicle_count", "vehicle_ids"]
    ]



def vehicle_summary_dataframe(
    route_df: pd.DataFrame,
    tasks: list[RequestTask],
    vehicle_states: list[VehicleState],
) -> pd.DataFrame:
    """실제로 확정된 비행 기록을 차량별로 집계합니다."""
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
        actual_drop_penalty = sum(
            task.drop_penalty for task in assigned_tasks if task.drop_penalty_applied
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
                    0 if flight_start is None or flight_end is None else flight_end - flight_start
                ),
                "fuel_shortage_count": fuel_shortage_count,
                "empty_movement_count": empty_count,
                "passenger_movement_count": passenger_count,
                "cumulative_passengers": cumulative_passengers,
                "average_onboard_passengers": round(average_onboard, 3),
                "penalty": int(assigned_request_penalty_score),
                "assigned_request_penalty_score": int(assigned_request_penalty_score),
                "actual_drop_penalty": int(actual_drop_penalty),
            }
        )

    return pd.DataFrame(rows)

def save_csv(df: pd.DataFrame, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")


# ============================================================
# 메인 Rolling Horizon 실행
# ============================================================
def main() -> None:
    start_clock = time.perf_counter()
    if (
        isinstance(TIME_LIMIT_SECONDS, bool)
        or not isinstance(TIME_LIMIT_SECONDS, int)
        or TIME_LIMIT_SECONDS < 1
    ):
        raise ValueError("TIME_LIMIT_SECONDS는 1 이상의 정수(초)여야 합니다.")

    distance_matrix = load_matrix(DISTANCE_MATRIX_PATH, integer=False)
    time_matrix = load_matrix(TIME_MATRIX_PATH, integer=True)
    nodes = load_nodes(NODE_REFERENCE_PATH)

    if set(distance_matrix.index) != set(time_matrix.index):
        raise ValueError("거리행렬과 시간행렬의 노드가 일치하지 않습니다.")
    if not set(distance_matrix.index).issubset(nodes):
        raise ValueError("행렬 노드 중 노드 참조표에 없는 노드가 있습니다.")

    depot_ids = [normalize_id(value) for value in DEPOT_ROUTE_NODE_IDS]
    missing_depots = [value for value in depot_ids if value not in nodes]
    if missing_depots:
        raise ValueError(f"Depot 노드가 참조표에 없습니다: {missing_depots}")

    tasks = load_tasks(REQUEST_PATH, distance_matrix, time_matrix, nodes)
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
    simulation_end = maximum_departure + int(time_matrix.to_numpy().max()) + SERVICE_TIME_MINUTES + 1

    for horizon_start in range(0, simulation_end + 1, REOPTIMIZATION_INTERVAL_MINUTES):
        horizon_end = horizon_start + ROLLING_HORIZON_MINUTES
        update_request_states(tasks, horizon_start)

        active_tasks = [
            task
            for task in tasks
            if task.status in {"pending", "planned"}
            and task.min_available_departure <= horizon_end
            and task.max_available_departure >= horizon_start
        ]

        objective_value: int | None = None
        planned_count = 0
        committed_count = 0
        solver_seconds = 0.0

        if active_tasks:
            model_start = time.perf_counter()
            model = build_horizon_model(
                active_tasks,
                vehicle_states,
                distance_matrix,
                time_matrix,
                horizon_start,
                horizon_end,
            )
            solution = solve_horizon(model)
            solver_seconds = time.perf_counter() - model_start
            route_plan, task_plan = extract_plan(model, solution, NUM_VEHICLES)
            objective_value = None if solution is None else int(solution.ObjectiveValue())

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
                    }
                )

            commit_end = horizon_start + REOPTIMIZATION_INTERVAL_MINUTES
            committed_count = commit_first_tasks(
                route_plan,
                task_lookup,
                vehicle_states,
                distance_matrix,
                time_matrix,
                nodes,
                horizon_start,
                commit_end,
                route_leg_rows,
                stop_rows,
            )

            # 미래 계획은 다음 Horizon에서 다시 계산하므로 확정되지 않은 planned를 pending으로 되돌립니다.
            for task in active_tasks:
                if task.status == "planned":
                    task.status = "pending"

        update_request_states(tasks, horizon_start + REOPTIMIZATION_INTERVAL_MINUTES)
        summary_rows.append(
            {
                "horizon_start": horizon_start,
                "horizon_end": horizon_end,
                "active_request_count": len(active_tasks),
                "planned_request_count": planned_count,
                "committed_request_count": committed_count,
                "completed_request_count": sum(task.status == "completed" for task in tasks),
                "onboard_request_count": sum(task.status == "onboard" for task in tasks),
                "overdue_request_count": sum(task.status == "overdue" for task in tasks),
                "drop_penalty_total": sum(
                    task.drop_penalty for task in tasks if task.drop_penalty_applied
                ),
                "objective_value": objective_value,
                "solver_runtime_seconds": round(solver_seconds, 3),
            }
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

    print("\n========== 실행 결과 ==========")
    print(f"요청 batch 수: {len(tasks):,}")
    print(f"완료: {sum(task.status == 'completed' for task in tasks):,}")
    print(f"운항 중: {sum(task.status == 'onboard' for task in tasks):,}")
    print(f"Overdue: {sum(task.status == 'overdue' for task in tasks):,}")
    print(f"Vehicle summary: {len(vehicle_summary_df):,}대")
    print(f"실행시간: {time.perf_counter() - start_clock:.2f}초")
    print(f"결과 폴더: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
