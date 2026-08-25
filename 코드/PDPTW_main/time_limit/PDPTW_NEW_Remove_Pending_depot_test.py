# -*- coding: utf-8 -*-
"""OR-Tools rolling-horizon PDPTW simulation.

Times are integer minutes from BASE_TIME.  Energy is represented by remaining
flight range in metres inside OR-Tools and by kilometres in reports/state.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_DIR / "자료" / "기초자료"
REQUEST_PATH = INPUT_DIR / "finalDemand_v5" / "finalDemand_v5" / "d5000_s01.csv"
DISTANCE_MATRIX_PATH = INPUT_DIR / "distance_matrix_km.csv"
TIME_MATRIX_PATH = INPUT_DIR / "flight_time_matrix_min_remove_fuel.csv"
NODE_REFERENCE_PATH = INPUT_DIR / "vp_reference.csv"
TRANSPORTATION_MATRIX_PATH = PROJECT_DIR / "자료" / "결과" / "차량_교통수단" /"public_transit_time_matrix_min.csv"
OUTPUT_DIR = PROJECT_DIR / "자료" / "결과" / "Ortools" / "d1000" / "Penalty_Per_Vehicles" / "Remove_Pending_Depot" / "Pending" / "100000"

BASE_TIME = "05:40"
END_TIME = "19:16"
NUM_VEHICLES = 180
VEHICLE_CAPACITY = 3
DEPOT_ROUTE_NODE_IDS = list(range(1, 11))
ROLLING_HORIZON_MINUTES = 30
REOPTIMIZATION_INTERVAL_MINUTES = 20
TIME_LIMIT_SECONDS = 120

SERVICE_TIME_MINUTES = 3
BOARDING_CHARGE_MINUTES = 2
TAXI_TIME_MINUTES = 1
BASE_DROP_PENALTY = 1_000_000
MAX_WAIT_PENALTY_WEIGHT = 1_000
TRANSPORTATION_JOBY_PENALTY_WEIGHT = 10_000
INITIAL_REMAINING_RANGE_KM = 160.0
MAX_REMAINING_RANGE_KM = 160.0
MIN_REMAINING_RANGE_KM = 15.0
LOW_RANGE_CHARGE_TARGET_KM = 100.0
CHARGING_RATE_KM_PER_MIN = 4.267
LOG_SEARCH = False
LOG_ROLLING_HORIZON = True

COLUMN_ALIASES = {
    "origin": ("n", "origin", "origin_node", "pickup_node"),
    "destination": ("m", "destination", "destination_node", "delivery_node"),
    "passengers": ("cnt", "passenger", "passengers", "demand"),
    "max_wait": ("maxWait_min_round", "maxWait_min", "max_wait_min"),
    "departure": ("art", "depature_time", "departure_time", "requested_departure_time"),
    "arrival": ("arrT", "arrival_time", "requested_arrival_time"),
    "transportation": ("Transportation_min", "movement_time", "ground_movement_time", "car_time"),
    "joby_time": ("Joby_min", "Joby_Flight_Time", "joby_flight_time", "uam_time"),
    "request_id": ("request_id", "alternative_id", "request"),
    "batch_id": ("batch_id", "passenger_batch_id", "demand_id"),
    "route_node": ("vp_id", "route_node_id", "matrix_node_id", "node_id"),
    "physical_node": ("Node", "physical_node_id", "physical_id"),
    "address": ("vpname", "physical_address", "address"),
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
    batch_id: str
    origin: str
    destination: str
    passengers: int
    art: int
    departure_time: int
    arrival_time: int
    max_wait_min: int
    transportation_min: int
    joby_min: int
    distance_km: float
    pending_count: int = 0
    status: str = "Pending"
    selected: bool = False
    assigned_vehicle_id: int | None = None
    pickup_time: int | None = None
    delivery_time: int | None = None

    @property
    def window_end(self) -> int:
        return self.arrival_time + self.max_wait_min


@dataclass
class BatchState:
    batch_id: str
    alternatives: list[RequestTask] = field(default_factory=list)
    status: str = "Pending"
    pending_count: int = 0
    selected_request_id: str | None = None
    assigned_vehicle_id: int | None = None
    pickup_time: int | None = None
    delivery_time: int | None = None


@dataclass
class VehicleState:
    vehicle_id: int
    home_depot: str
    route_node_id: str
    node_arrival_time: int = 0
    available_time: int = 0
    remaining_range_km: float = INITIAL_REMAINING_RANGE_KM
    onboard_batches: dict[str, int] = field(default_factory=dict)
    committed_deliveries: dict[str, str] = field(default_factory=dict)


@dataclass
class HorizonModel:
    manager: Any
    routing: Any
    time_dimension: Any
    range_dimension: Any
    node_meta: dict[int, tuple[str, RequestTask | None]]
    pickup_nodes: dict[str, int]
    delivery_nodes: dict[str, int]


def normalize_id(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        return str(int(number)) if number.is_integer() else text
    except ValueError:
        return text


def find_column(df: pd.DataFrame, key: str, required: bool = True) -> str | None:
    names = {str(c).strip().lower(): str(c) for c in df.columns}
    for alias in COLUMN_ALIASES[key]:
        if alias.lower() in names:
            return names[alias.lower()]
    if required:
        raise KeyError(f"필수 열 '{key}'이 없습니다. 허용 alias={COLUMN_ALIASES[key]}, 실제 열={list(df.columns)}")
    return None


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"CSV encoding을 판별할 수 없습니다: {path}")


def load_matrix(path: Path, integer: bool) -> pd.DataFrame:
    df = read_table(path).set_index(read_table(path).columns[0])
    df.index = [normalize_id(v) for v in df.index]
    df.columns = [normalize_id(v) for v in df.columns]
    df = df.apply(pd.to_numeric, errors="raise")
    if set(df.index) != set(df.columns):
        raise ValueError(f"행렬의 행/열 node가 일치하지 않습니다: {path}")
    df = df.loc[df.index, df.index]
    return df.round().astype(int) if integer else df.astype(float)


def load_nodes(path: Path) -> dict[str, NodeInfo]:
    df = read_table(path)
    rc = find_column(df, "route_node")
    pc = find_column(df, "physical_node", False) or rc
    ac = find_column(df, "address", False)
    xc, yc = find_column(df, "longitude"), find_column(df, "latitude")
    return {normalize_id(r[rc]): NodeInfo(normalize_id(r[rc]), normalize_id(r[pc]), "" if ac is None else str(r[ac]), float(r[xc]), float(r[yc])) for _, r in df.iterrows()}

# 각 지역에 대한 딕셔너리 생성
def load_named_matrix(path: Path, nodes: dict[str, NodeInfo]) -> pd.DataFrame:
    """Load a matrix whose axes use vp names and convert them to route-node IDs."""
    df = read_table(path)
    df = df.set_index(df.columns[0])
    name_to_id = {node.address.strip(): node_id for node_id, node in nodes.items()}
    unknown_rows = [str(v) for v in df.index if str(v).strip() not in name_to_id]
    unknown_cols = [str(v) for v in df.columns if str(v).strip() not in name_to_id]
    if unknown_rows or unknown_cols:
        raise ValueError(f"Transportation matrix의 vertiport 이름이 vp_reference와 다릅니다: rows={unknown_rows}, cols={unknown_cols}")
    df.index = [name_to_id[str(v).strip()] for v in df.index]
    df.columns = [name_to_id[str(v).strip()] for v in df.columns]
    return df.apply(pd.to_numeric, errors="raise").astype(float)

# ortools의 time의 변형
def base_time_minutes() -> int:
    h, m = map(int, BASE_TIME.split(":")[:2])
    return h * 60 + m


def end_time_minutes() -> int:
    h, m = map(int, END_TIME.split(":")[:2])
    result = h * 60 + m - base_time_minutes()
    return result + 1440 if result < 0 else result


def to_offset_minutes(value: object) -> int:
    if pd.isna(value):
        raise ValueError("arrival/departure time 값이 비어 있습니다")
    if isinstance(value, (int, float)): # float인 경우, int로 변형. 단, 본 data는 전부 분의 형태로 구성 돼 있으므로 문제 X
        return int(round(float(value)))
    parts = str(value).strip().split(":")
    if len(parts) < 2:
        return int(round(float(value)))
    result = int(parts[0]) * 60 + int(parts[1]) - base_time_minutes()
    return result + 1440 if result < 0 else result


def drop_penalty(batch: BatchState, maximum_max_wait: int) -> int:
    representative = batch.alternatives[0]
    flight_time_matrix = getattr(drop_penalty, "_flight_time_matrix", None)
    if flight_time_matrix is None:
        flight_time_matrix = load_matrix(TIME_MATRIX_PATH, False)
        setattr(drop_penalty, "_flight_time_matrix", flight_time_matrix)
    joby_min = float(flight_time_matrix.loc[representative.origin, representative.destination])
    return int(BASE_DROP_PENALTY + MAX_WAIT_PENALTY_WEIGHT * (maximum_max_wait - representative.max_wait_min) + TRANSPORTATION_JOBY_PENALTY_WEIGHT * (representative.transportation_min - joby_min))


def load_tasks(request_path: Path, distance: pd.DataFrame, flight_time: pd.DataFrame, nodes: dict[str, NodeInfo], transportation_matrix: pd.DataFrame | None = None) -> tuple[list[RequestTask], dict[str, BatchState]]:
    raw = read_table(request_path)
    oc, dc = find_column(raw, "origin"), find_column(raw, "destination")
    cc, wc = find_column(raw, "passengers"), find_column(raw, "max_wait")
    depc = find_column(raw, "departure", False)
    arrc = find_column(raw, "arrival", False)
    tc, jc = find_column(raw, "transportation", False), find_column(raw, "joby_time", False)
    ric, bic = find_column(raw, "request_id", False), find_column(raw, "batch_id", False)
    tasks: list[RequestTask] = []
    for pos, (_, row) in enumerate(raw.iterrows()):
        origin, destination = normalize_id(row[oc]), normalize_id(row[dc])
        if origin == destination:
            continue
        if origin not in nodes or destination not in nodes or origin not in distance.index or destination not in distance.columns:
            raise KeyError(f"수요 row {pos}의 node가 행렬/참조에 없습니다: {origin}->{destination}")
        if depc:
            art = to_offset_minutes(row[depc])
        elif {"depT_hr", "depT_m"}.issubset(raw.columns):
            art = int(row["depT_hr"]) * 60 + int(row["depT_m"]) - base_time_minutes()
        else:
            raise KeyError("필수 arrival time 열(art/departure 또는 depT_hr+depT_m)이 없습니다")
        departure_time = max(0, art)
        if arrc:
            arrival_time = to_offset_minutes(row[arrc])
        elif {"arrT_hr", "arrT_m"}.issubset(raw.columns):
            arrival_time = int(row["arrT_hr"]) * 60 + int(row["arrT_m"]) - base_time_minutes()
        else:
            raise KeyError("필수 arrival time 열(arrT/arrival_time 또는 arrT_hr+arrT_m)이 없습니다")
        if art < 0 or art > end_time_minutes() or arrival_time < 0 or arrival_time > end_time_minutes():
            continue
        if jc:
            joby = int(math.ceil(float(row[jc])))
        elif {"arrT_hr", "arrT_m", "depT_hr", "depT_m"}.issubset(raw.columns):
            joby = (int(row["arrT_hr"]) * 60 + int(row["arrT_m"])) - (int(row["depT_hr"]) * 60 + int(row["depT_m"]))
        else:
            joby = int(flight_time.loc[origin, destination])
        if tc:
            transportation = int(math.ceil(float(row[tc])))
        elif transportation_matrix is not None:
            transportation = int(math.ceil(float(transportation_matrix.loc[origin, destination])))
        else:
            raise KeyError("필수 Transportation_min 열(또는 depT_hr/depT_m/arrT_hr/arrT_m)이 없습니다")
        if transportation <= joby:
            raise ValueError(f"row {pos}: Transportation_min({transportation})은 Joby_min({joby})보다 커야 합니다")
        request_id = normalize_id(row[ric]) if ric else f"R{pos + 1}"
        batch_id = normalize_id(row[bic]) if bic else f"B{pos + 1}"
        count = int(math.ceil(float(row[cc])))
        if count < 1:
            raise ValueError(f"row {pos}: cnt는 1 이상이어야 합니다: {count}")
        if bic and count > VEHICLE_CAPACITY:
            raise ValueError(f"row {pos}: 명시적 batch_id의 cnt={count}가 vehicle capacity={VEHICLE_CAPACITY}를 초과합니다")
        # Legacy aggregate-demand CSVs have no batch_id. Preserve their meaning by
        # splitting cnt into capacity-sized, independent passenger batches.
        parts = int(math.ceil(count / VEHICLE_CAPACITY))
        remaining = count
        # Passenger batch_id를 생성
        for part in range(parts):
            passengers = min(VEHICLE_CAPACITY, remaining)
            part_batch = batch_id if parts == 1 else f"{batch_id}-{part + 1}"
            part_request = request_id if parts == 1 else f"{request_id}-{part + 1}"
            tasks.append(RequestTask(f"{part_batch}:{part_request}", part_request, part_batch, origin, destination, passengers, max(0, art), departure_time, arrival_time, int(math.ceil(float(row[wc]))), transportation, joby, float(distance.loc[origin, destination])))
            remaining -= passengers
    batches: dict[str, BatchState] = {}
    for task in tasks:
        batches.setdefault(task.batch_id, BatchState(task.batch_id)).alternatives.append(task)
    return tasks, batches


"""변경 시작: 중간 Rolling Horizon의 End를 Open End로 처리하기 위한 설정"""
def build_horizon_model(active: list[RequestTask], batches: dict[str, BatchState], vehicles: list[VehicleState], distance: pd.DataFrame, flight_time: pd.DataFrame, hs: int, he: int, maximum_max_wait: int, return_to_home: bool = False) -> HorizonModel:
    """변경 끝"""
    vehicle_count = len(vehicles)
    starts = list(range(vehicle_count))
    ends = list(range(vehicle_count, vehicle_count * 2))
    node_meta: dict[int, tuple[str, RequestTask | None]] = {}
    pickup_nodes, delivery_nodes = {}, {}
    cursor = vehicle_count * 2
    for task in active:
        pickup_nodes[task.task_key] = cursor
        delivery_nodes[task.task_key] = cursor + 1
        node_meta[cursor], node_meta[cursor + 1] = ("Pickup", task), ("Delivery", task)
        cursor += 2
    manager = pywrapcp.RoutingIndexManager(cursor, vehicle_count, starts, ends)
    routing = pywrapcp.RoutingModel(manager)
    by_vehicle = {v.vehicle_id: v for v in vehicles}

    def location(model_node: int) -> str:
        # 이번 RH 시작 시 차량의 현재 위치
        if model_node < vehicle_count:
            return by_vehicle[model_node].route_node_id
        # 차량의 원래 Depot
        if model_node < vehicle_count * 2:
            return by_vehicle[model_node - vehicle_count].home_depot
        # Request Node
        event, task = node_meta[model_node]
        return task.origin if event == "Pickup" else task.destination  # type: ignore[union-attr]

    def service(model_node: int) -> int:
        return SERVICE_TIME_MINUTES if model_node in node_meta else 0 #어차피 모든 노드는 전부 PD 노드 그래서 추후에 진행하는 SERVICE_TIME_MINUTES를 진행.

    """변경 시작: 중간 Rolling Horizon에서는 End까지의 가상 이동 비용을 0으로 처리"""
    def time_cb(fi: int, ti: int) -> int:
        f, t = manager.IndexToNode(fi), manager.IndexToNode(ti)

        # [수정] Open End의 가상 End 이동거리는 0이지만,
        # 마지막 Pickup/Delivery에서 발생한 SERVICE_TIME_MINUTES는 반영
        if not return_to_home and vehicle_count <= t < vehicle_count * 2:
            return service(f)

        # [수정] 일반 이동은 이전 노드의 Service Time + 실제 비행시간
        return service(f) + int(
            flight_time.loc[location(f), location(t)]
        )

    def distance_cb(fi: int, ti: int) -> int:
        f, t = manager.IndexToNode(fi), manager.IndexToNode(ti)
        if not return_to_home and vehicle_count <= t < vehicle_count * 2: # 중간노드는 아예 페널티 계산 X
            return 0
        return int(round(float(distance.loc[location(f), location(t)]) * 1000))
    """변경 끝"""

    def range_cb(fi: int, ti: int) -> int: # 각 경로별 거리 출력 이 거리를 통해 vehicle의 잔여 비행가능 거리를 업데이트 
        return -distance_cb(fi, ti) # 각 경로별 거리 출력 이 거리를 통해 vehicle의 잔여 비행가능 거리를 업데이트 

    ti = routing.RegisterTransitCallback(time_cb)
    di = routing.RegisterTransitCallback(distance_cb) # 경로별 거리
    ri = routing.RegisterTransitCallback(range_cb) # 비행기 잔여 비행가능 거리 
    routing.SetArcCostEvaluatorOfAllVehicles(di)
    model_end = he + int(flight_time.to_numpy().max()) * 4 + SERVICE_TIME_MINUTES * 4
    routing.AddDimension(ti, model_end, model_end, False, "Time")
    td = routing.GetDimensionOrDie("Time")
    max_range_m, reserve_m = int(MAX_REMAINING_RANGE_KM * 1000), int(MIN_REMAINING_RANGE_KM * 1000)
    routing.AddDimension(ri, max_range_m, max_range_m, False, "RemainingRange")
    rd = routing.GetDimensionOrDie("RemainingRange")
    solver = routing.solver()
    charge_rate_m = int(round(CHARGING_RATE_KM_PER_MIN * 1000))

    for v in vehicles:
        si, ei = routing.Start(v.vehicle_id), routing.End(v.vehicle_id)
        start_time = max(hs, v.available_time)
        td.CumulVar(si).SetValue(start_time)
        td.CumulVar(ei).SetRange(start_time, model_end)
        rd.CumulVar(si).SetValue(int(round(v.remaining_range_km * 1000)))
        rd.CumulVar(ei).SetRange(reserve_m, max_range_m)
        routing.AddVariableMinimizedByFinalizer(td.CumulVar(ei))

    by_batch: dict[str, list[int]] = {}
    for task in active:
        pnode, dnode = pickup_nodes[task.task_key], delivery_nodes[task.task_key]
        pi, deli = manager.NodeToIndex(pnode), manager.NodeToIndex(dnode)
        td.CumulVar(pi).SetRange(max(hs, task.departure_time), he)
        td.CumulVar(deli).SetRange(max(hs, task.arrival_time), min(task.window_end, end_time_minutes()))
        routing.AddPickupAndDelivery(pi, deli)
        solver.Add(routing.VehicleVar(pi) == routing.VehicleVar(deli))
        solver.Add(td.CumulVar(pi) <= td.CumulVar(deli))
        solver.Add(routing.ActiveVar(pi) == routing.ActiveVar(deli))
        routing.AddDisjunction([deli], 0)
        by_batch.setdefault(task.batch_id, []).append(pi)
        for idx in (pi, deli):
            rd.CumulVar(idx).SetRange(reserve_m, max_range_m)
            rd.SlackVar(idx).SetRange(0, max_range_m)
            solver.Add(rd.SlackVar(idx) <= charge_rate_m * (td.SlackVar(idx) + BOARDING_CHARGE_MINUTES))
            solver.Add(rd.CumulVar(idx) + rd.SlackVar(idx) <= max_range_m)

    for batch_id, pickup_indices in by_batch.items():
        penalty = drop_penalty(batches[batch_id], maximum_max_wait)
        routing.AddDisjunction(pickup_indices, penalty, 1)

    demands = [0] * cursor
    for node, (event, task) in node_meta.items():
        demands[node] = task.passengers if event == "Pickup" else -task.passengers  # type: ignore[union-attr]
    demand_idx = routing.RegisterUnaryTransitCallback(lambda index: demands[manager.IndexToNode(index)])
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [VEHICLE_CAPACITY] * vehicle_count, True, "Capacity")
    return HorizonModel(manager, routing, td, rd, node_meta, pickup_nodes, delivery_nodes)


def solve_horizon(model: HorizonModel):
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromSeconds(TIME_LIMIT_SECONDS)
    p.log_search = LOG_SEARCH
    return model.routing.SolveWithParameters(p)


def extract_routes(model: HorizonModel, solution: Any, vehicles: list[VehicleState]) -> tuple[list[dict[str, Any]], set[str]]:
    rows, selected = [], set()
    if solution is None:
        return rows, selected
    for v in vehicles:
        index, sequence = model.routing.Start(v.vehicle_id), 0
        while not model.routing.IsEnd(index):
            nxt = solution.Value(model.routing.NextVar(index))
            node = model.manager.IndexToNode(nxt)
            if node in model.node_meta:
                event, task = model.node_meta[node]
                rows.append({"vehicle_id": v.vehicle_id, "sequence": sequence, "event": event, "task": task, "time": int(solution.Value(model.time_dimension.CumulVar(nxt)))})
                if event == "Pickup":
                    selected.add(task.task_key)  # type: ignore[union-attr]
                sequence += 1
            index = nxt
    return rows, selected

# 비행 가능 조건을 추출.
def charge_for_departure(current_range: float, distance_km: float) -> tuple[int, float]:
    if current_range - distance_km > MIN_REMAINING_RANGE_KM:
        return 0, current_range
    target = max(LOW_RANGE_CHARGE_TARGET_KM, distance_km + MIN_REMAINING_RANGE_KM + 0.001)
    if target > MAX_REMAINING_RANGE_KM + 1e-9:
        raise RuntimeError(f"160km range로 reserve를 지킬 수 없는 leg입니다: distance={distance_km:.3f}km")
    minutes = int(math.ceil(max(0.0, target - current_range) / CHARGING_RATE_KM_PER_MIN))
    return minutes, min(MAX_REMAINING_RANGE_KM, current_range + minutes * CHARGING_RATE_KM_PER_MIN)

# 각 구간마다 진행할 예정의 건수를 출력
def commit_routes(route_events: list[dict[str, Any]], vehicles: list[VehicleState], distance: pd.DataFrame, flight_time: pd.DataFrame, nodes: dict[str, NodeInfo], commit_end: int, log_rows: list[dict[str, Any]], batches: dict[str, BatchState]) -> int:
    state_by_id = {v.vehicle_id: v for v in vehicles}
    committed = 0
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in route_events:
        grouped.setdefault(row["vehicle_id"], []).append(row)
    for vehicle_id, events in grouped.items():
        state = state_by_id[vehicle_id]
        chosen = {r["task"].task_key for r in events if r["event"] == "Pickup" and r["time"] < commit_end}
        if not chosen:
            continue
        projected_node, projected_time, projected_remaining = state.route_node_id, state.available_time, state.remaining_range_km
        late_task_keys: set[str] = set()
        for projected_row in [r for r in events if r["task"].task_key in chosen]:
            projected_task, projected_event = projected_row["task"], projected_row["event"]
            projected_target = projected_task.origin if projected_event == "Pickup" else projected_task.destination
            projected_dist, projected_travel = float(distance.loc[projected_node, projected_target]), int(flight_time.loc[projected_node, projected_target])
            projected_planned = max(projected_time, projected_row["time"])
            projected_parking = max(0, projected_planned - projected_time - projected_travel)
            projected_remaining = min(MAX_REMAINING_RANGE_KM, projected_remaining + projected_parking * CHARGING_RATE_KM_PER_MIN)
            projected_extra_charge, projected_departure_range = charge_for_departure(projected_remaining, projected_dist)
            projected_departure = projected_time + projected_parking + projected_extra_charge
            projected_arrival = projected_departure + projected_travel
            projected_after_flight = projected_departure_range - projected_dist
            projected_service_charge = min(MAX_REMAINING_RANGE_KM, projected_after_flight + BOARDING_CHARGE_MINUTES * CHARGING_RATE_KM_PER_MIN)
            if projected_event == "Delivery" and projected_arrival > end_time_minutes():
                late_task_keys.add(projected_task.task_key)
            projected_node, projected_time, projected_remaining = projected_target, projected_arrival + SERVICE_TIME_MINUTES, projected_service_charge
        chosen -= late_task_keys
        if not chosen:
            continue
        # Commit the complete pickup/delivery sequence for pickups inside the frozen interval.
        events = [r for r in events if r["task"].task_key in chosen]
        onboard = dict(state.onboard_batches)
        current_node, current_time, remaining = state.route_node_id, state.available_time, state.remaining_range_km
        for row in events:
            task, event = row["task"], row["event"]
            target = task.origin if event == "Pickup" else task.destination
            dist, travel = float(distance.loc[current_node, target]), int(flight_time.loc[current_node, target])
            planned = max(current_time, row["time"])
            parking = max(0, planned - current_time - travel)
            """"""  # 변경 시작: 출발 전 waiting/charging을 Movement와 분리한다. Movement 중에는 충전하지 않으며, 동일 노드 이동은 실제 비행으로 기록하지 않는다.
            range_before_stay = remaining
            remaining = min(MAX_REMAINING_RANGE_KM, remaining + parking * CHARGING_RATE_KM_PER_MIN)
            extra_charge, departure_range = charge_for_departure(remaining, dist)
            departure = current_time + parking + extra_charge
            arrival = departure + travel
            after_flight = departure_range - dist
            before_count = sum(onboard.values())
            related = task.batch_id

            if parking + extra_charge > 0:
                log_rows.append({"vehicle_id": vehicle_id, "sequence": len(log_rows), "event": "Stay", "from_node": current_node, "to_node": current_node, "physical_from_node": nodes[current_node].physical_node_id, "physical_to_node": nodes[current_node].physical_node_id, "departure_time": departure, "arrival_time": current_time, "travel_time": 0, "travel_distance_km": 0.0, "charging_time": parking + extra_charge, "charged_range_km": round(departure_range - range_before_stay, 3), "remaining_range_before": round(range_before_stay, 3), "remaining_range_after": round(departure_range, 3), "onboard_passengers": before_count, "related_batch_id": related, "movement_type": "parking_charging", "from_event": "Position", "to_event": "Movement" if current_node != target else event})

            if current_node != target:
                log_rows.append({"vehicle_id": vehicle_id, "sequence": len(log_rows), "event": "Movement", "from_node": current_node, "to_node": target, "physical_from_node": nodes[current_node].physical_node_id, "physical_to_node": nodes[target].physical_node_id, "departure_time": departure, "arrival_time": arrival, "travel_time": travel, "travel_distance_km": round(dist, 3), "charging_time": 0, "charged_range_km": 0.0, "remaining_range_before": round(departure_range, 3), "remaining_range_after": round(after_flight, 3), "onboard_passengers": before_count, "related_batch_id": related, "movement_type": "passenger_flight" if before_count else "empty_repositioning", "from_event": "Position", "to_event": event})
            """"""  # 변경 끝: waiting/charging은 Stay에 귀속하고, Movement는 순수 비행만 기록한다. 따라서 Movement의 charging_time/charged_range_km는 항상 0이다.
            service_start = arrival
            service_charge = min(MAX_REMAINING_RANGE_KM, after_flight + BOARDING_CHARGE_MINUTES * CHARGING_RATE_KM_PER_MIN)
            if event == "Pickup":
                onboard[task.batch_id] = task.passengers
                task.pickup_time, task.assigned_vehicle_id, task.selected = service_start, vehicle_id, True
                batch = batches[task.batch_id]
                batch.selected_request_id, batch.assigned_vehicle_id, batch.pickup_time = task.request_id, vehicle_id, service_start
                committed += 1
            else:
                onboard.pop(task.batch_id, None)
                task.delivery_time, task.status = service_start, "Complete"
                batch = batches[task.batch_id]
                batch.delivery_time, batch.status = service_start, "Complete"
            log_rows.append({"vehicle_id": vehicle_id, "sequence": len(log_rows), "event": event, "from_node": target, "to_node": target, "physical_from_node": nodes[target].physical_node_id, "physical_to_node": nodes[target].physical_node_id, "departure_time": service_start + SERVICE_TIME_MINUTES, "arrival_time": service_start, "travel_time": 0, "travel_distance_km": 0.0, "charging_time": BOARDING_CHARGE_MINUTES, "charged_range_km": round(service_charge - after_flight, 3), "remaining_range_before": round(after_flight, 3), "remaining_range_after": round(service_charge, 3), "onboard_passengers": sum(onboard.values()), "related_batch_id": related, "movement_type": event.lower() + "_service", "from_event": event, "to_event": event})
            current_node, current_time, remaining = target, service_start + SERVICE_TIME_MINUTES, service_charge
        state.route_node_id, state.node_arrival_time, state.available_time = current_node, current_time - SERVICE_TIME_MINUTES, current_time
        state.remaining_range_km, state.onboard_batches = remaining, onboard
    return committed

# completed, overdue의 여부에 따라 Node 상태를 업데이트 참고로 GPT의 트롤로 인해 batch == request OD 그룹들을 의미.
def update_batch_states(batches: dict[str, BatchState], current_time: int, increment_pending: bool = False, attempted_batch_ids: set[str] | None = None) -> None:
    for batch in batches.values():
        if batch.status == "Complete":
            continue
        future = [t for t in batch.alternatives if t.window_end >= current_time]
        if not future:
            batch.status = "Overdue"
            for task in batch.alternatives:
                if task.status != "Complete":
                    task.status = "Overdue"
        elif increment_pending and (attempted_batch_ids is None or batch.batch_id in attempted_batch_ids):
            batch.status = "Pending" # Status는 유지. 이전 RH에서 Request 구간 몇 개가 누락됐는지 알 수 있으므로.
            batch.pending_count += 1
            for task in future:
                task.pending_count = batch.pending_count


def return_to_depots(vehicles: list[VehicleState], distance: pd.DataFrame, flight_time: pd.DataFrame, nodes: dict[str, NodeInfo], logs: list[dict[str, Any]]) -> None:
    for state in vehicles:
        if state.route_node_id == state.home_depot:
            continue
        dist = float(distance.loc[state.route_node_id, state.home_depot])
        charge_min, departure_range = charge_for_departure(state.remaining_range_km, dist)
        departure, arrival = state.available_time + charge_min, state.available_time + charge_min + int(flight_time.loc[state.route_node_id, state.home_depot])
        """"""  # 변경 시작: 최종 Depot 복귀 전 필요한 충전도 Movement가 아니라 출발 노드의 Stay로 분리한다.
        if charge_min > 0:
            logs.append({"vehicle_id": state.vehicle_id, "sequence": len(logs), "event": "Stay", "from_node": state.route_node_id, "to_node": state.route_node_id, "physical_from_node": nodes[state.route_node_id].physical_node_id, "physical_to_node": nodes[state.route_node_id].physical_node_id, "departure_time": departure, "arrival_time": state.available_time, "travel_time": 0, "travel_distance_km": 0.0, "charging_time": charge_min, "charged_range_km": round(departure_range - state.remaining_range_km, 3), "remaining_range_before": round(state.remaining_range_km, 3), "remaining_range_after": round(departure_range, 3), "onboard_passengers": 0, "related_batch_id": "", "movement_type": "parking_charging", "from_event": "Position", "to_event": "Movement"})
        logs.append({"vehicle_id": state.vehicle_id, "sequence": len(logs), "event": "Movement", "from_node": state.route_node_id, "to_node": state.home_depot, "physical_from_node": nodes[state.route_node_id].physical_node_id, "physical_to_node": nodes[state.home_depot].physical_node_id, "departure_time": departure, "arrival_time": arrival, "travel_time": arrival - departure, "travel_distance_km": round(dist, 3), "charging_time": 0, "charged_range_km": 0.0, "remaining_range_before": round(departure_range, 3), "remaining_range_after": round(departure_range - dist, 3), "onboard_passengers": 0, "related_batch_id": "", "movement_type": "return_to_depot", "from_event": "Position", "to_event": "Depot"})
        """"""  # 변경 끝: return_to_depot Movement 역시 순수 비행만 기록하며 충전 정보는 직전 Stay에 기록한다.
        state.route_node_id, state.available_time, state.node_arrival_time, state.remaining_range_km = state.home_depot, arrival, arrival, departure_range - dist


def format_hhmm(value: int | None) -> str:
    if value is None:
        return ""
    absolute = base_time_minutes() + int(value)
    return f"{(absolute // 60) % 24:02d}:{absolute % 60:02d}"


def batch_dataframe(batches: dict[str, BatchState], maximum_max_wait: int) -> pd.DataFrame:
    rows = []
    for batch in batches.values():
        selected = next((t for t in batch.alternatives if t.request_id == batch.selected_request_id), batch.alternatives[0])
        # [추가 수정 1] rolling_horizon_request_status.csv에 각 Request의 원래 Delivery Time Window를 분 단위와 HH:MM 형식으로 함께 출력한다.
        rows.append({"batch_id": batch.batch_id, "selected_request_id": batch.selected_request_id, "origin": selected.origin, "destination": selected.destination, "max_wait": selected.max_wait_min, "time_window_start": selected.arrival_time, "time_window_start_hhmm": format_hhmm(selected.arrival_time), "time_window_end": selected.window_end, "time_window_end_hhmm": format_hhmm(selected.window_end), "Transportation_min": selected.transportation_min, "Joby_min": selected.joby_min, "Transportation_Joby_time_saving": selected.transportation_min - selected.joby_min, "pending_count": batch.pending_count, "final_drop_penalty": drop_penalty(batch, maximum_max_wait), "assigned_vehicle": batch.assigned_vehicle_id, "pickup_time": batch.pickup_time, "pickup_time_hhmm": format_hhmm(batch.pickup_time), "delivery_time": batch.delivery_time, "delivery_time_hhmm": format_hhmm(batch.delivery_time), "final_status": batch.status})
        # [추가 수정 1 끝] time_window_start=arrival_time, time_window_end=arrival_time+max_wait이며 기존 계산/상태/패널티 로직은 변경하지 않는다.
    return pd.DataFrame(rows)


def occupancy_dataframe(route_df: pd.DataFrame, nodes: dict[str, NodeInfo], vehicles: list[VehicleState]) -> pd.DataFrame:
    """Preserve the minute-level vertiport occupancy output using each vehicle's physical location."""
    # [추가 수정 3] 각 Vehicle을 t=0의 home_depot에서 시작시키고 Movement 구간만 비행 중으로 제외하여,
    # Stay / Pickup / Delivery / 별도 로그가 없는 단순 대기까지 포함한 실제 노드별 물리적 차량 수를 계산한다.
    columns = ["time_min", "time_hhmm", "route_node_id", "physical_node_id", "physical_address", "vehicle_count", "vehicle_ids"]

    if not vehicles:
        return pd.DataFrame(columns=columns)

    if route_df.empty:
        end_time = 0
    else:
        end_time = int(max(route_df["arrival_time"].max(), route_df["departure_time"].max()))

    if vehicles:
        end_time = max(end_time, max(int(v.available_time) for v in vehicles))

    rows: list[dict[str, Any]] = []

    for state in vehicles:
        current_node = state.home_depot
        current_time = 0

        if route_df.empty:
            movements = pd.DataFrame()
        else:
            movements = route_df[
                (route_df["vehicle_id"] == state.vehicle_id)
                & (route_df["event"] == "Movement")
            ].sort_values(["departure_time", "arrival_time"])

        for _, movement in movements.iterrows():
            departure = int(movement["departure_time"])
            arrival = int(movement["arrival_time"])

            # 출발 직전까지는 현재 노드에 물리적으로 존재한다.
            for minute in range(current_time, departure):
                rows.append({
                    "time_min": minute,
                    "route_node_id": str(current_node),
                    "vehicle_id": state.vehicle_id,
                })

            # [departure, arrival) 동안은 비행 중이므로 어떤 노드에도 포함하지 않는다.
            current_node = str(movement["to_node"])
            current_time = arrival

        # 마지막 도착 이후 시뮬레이션 종료 시각까지 현재 노드에 계속 존재한다.
        for minute in range(current_time, end_time + 1):
            rows.append({
                "time_min": minute,
                "route_node_id": str(current_node),
                "vehicle_id": state.vehicle_id,
            })

    if not rows:
        return pd.DataFrame(columns=columns)

    result = (
        pd.DataFrame(rows)
        .groupby(["time_min", "route_node_id"], as_index=False)
        .agg(
            vehicle_count=("vehicle_id", "nunique"),
            vehicle_ids=("vehicle_id", lambda values: ",".join(map(str, sorted(set(values))))),
        )
    )
    result["time_hhmm"] = result["time_min"].map(format_hhmm)
    result["physical_node_id"] = result["route_node_id"].map(lambda value: nodes[value].physical_node_id)
    result["physical_address"] = result["route_node_id"].map(lambda value: nodes[value].address)
    # [추가 수정 3 끝] 기존 CSV 열 구조는 유지하고 vehicle_count의 의미만 '서비스 중 차량 수'에서
    # '해당 시각에 해당 노드에 실제로 존재하는 차량 수'로 변경한다.
    return result[columns]


def vehicle_summary_dataframe(route_df: pd.DataFrame, vehicles: list[VehicleState]) -> pd.DataFrame:
    rows = []
    for state in vehicles:
        vr = route_df[route_df["vehicle_id"] == state.vehicle_id] if not route_df.empty else pd.DataFrame()
        movement = vr[vr["event"] == "Movement"] if not vr.empty else pd.DataFrame()
        passenger = movement[movement["movement_type"] == "passenger_flight"] if not movement.empty else pd.DataFrame()
        empty = movement[movement["movement_type"].isin(["empty_repositioning", "return_to_depot"])] if not movement.empty else pd.DataFrame()
        """"""  # 변경 시작: vehicle_summary.csv에 실제 Movement 횟수와 그중 승객 탑승 passenger_flight 횟수를 추가한다.
        rows.append({"vehicle_id": state.vehicle_id, "home_depot": state.home_depot, "final_node": state.route_node_id, "final_available_time": state.available_time, "final_remaining_range_km": round(state.remaining_range_km, 3), "Movement": int(len(movement)) if not movement.empty else 0, "Passenger": int(len(passenger)) if not passenger.empty else 0, "total_flight_time_min": int(movement["travel_time"].sum()) if not movement.empty else 0, "total_flight_distance_km": round(float(movement["travel_distance_km"].sum()), 3) if not movement.empty else 0.0, "passenger_flight_distance_km": round(float(passenger["travel_distance_km"].sum()), 3) if not passenger.empty else 0.0, "empty_flight_distance_km": round(float(empty["travel_distance_km"].sum()), 3) if not empty.empty else 0.0, "charging_time_min": int(vr["charging_time"].sum()) if not vr.empty else 0, "pickup_count": int((vr["event"] == "Pickup").sum()) if not vr.empty else 0, "delivery_count": int((vr["event"] == "Delivery").sum()) if not vr.empty else 0})
        """"""  # 변경 끝: Movement는 전체 실제 비행 Movement 수, Passenger는 movement_type이 passenger_flight인 Movement 수이다.
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")


def main() -> None:
    started = time.perf_counter()
    distance, flight_time, nodes = load_matrix(DISTANCE_MATRIX_PATH, False), load_matrix(TIME_MATRIX_PATH, True), load_nodes(NODE_REFERENCE_PATH)
    transportation_matrix = load_named_matrix(TRANSPORTATION_MATRIX_PATH, nodes)
    if set(distance.index) != set(flight_time.index) or not set(distance.index).issubset(nodes):
        raise ValueError("distance/time/node-reference의 node 집합이 일치하지 않습니다")
    depots = [normalize_id(v) for v in DEPOT_ROUTE_NODE_IDS]
    missing = [v for v in depots if v not in distance.index]
    if missing:
        raise ValueError(f"Depot node가 행렬에 없습니다: {missing}")
    tasks, batches = load_tasks(REQUEST_PATH, distance, flight_time, nodes, transportation_matrix)
    maximum_max_wait = max(t.max_wait_min for t in tasks)
    """변경 시작: 마지막 Rolling Horizon에서 최초 home_depot으로 복귀하기 위한 기준 시각"""
    latest_window_end = max(t.window_end for t in tasks)
    """변경 끝"""
    vehicles = [VehicleState(i, depots[i % len(depots)], depots[i % len(depots)]) for i in range(NUM_VEHICLES)]
    logs, plans, summaries = [], [], []
    simulation_end = end_time_minutes()
    horizon_starts = range(0, simulation_end, REOPTIMIZATION_INTERVAL_MINUTES)
    total_horizons = len(horizon_starts)
    seen_batch_ids: set[str] = set()

    for rh_index, hs in enumerate(horizon_starts, start=1):
        update_batch_states(batches, hs)
        he = min(hs + ROLLING_HORIZON_MINUTES, simulation_end)
        commit_end = min(hs + REOPTIMIZATION_INTERVAL_MINUTES, simulation_end)
        active = [t for t in tasks if batches[t.batch_id].status == "Pending" and t.art <= he and t.window_end >= hs]
        active_batch_ids = {t.batch_id for t in active}
        new_batch_ids = active_batch_ids - seen_batch_ids
        active_pending_batch_ids = {batch_id for batch_id in active_batch_ids if batches[batch_id].pending_count > 0}
        onboard_batch_count = sum(len(v.onboard_batches) for v in vehicles)
        onboard_passenger_count = sum(sum(v.onboard_batches.values()) for v in vehicles)
        complete_before = sum(b.status == "Complete" for b in batches.values())
        overdue_before = sum(b.status == "Overdue" for b in batches.values())

        if LOG_ROLLING_HORIZON:
            progress = 100.0 * rh_index / total_horizons
            overlap_start = commit_end
            overlap_end = he
            print("\n" + "=" * 72, flush=True)
            print(f"[Rolling Horizon {rh_index}/{total_horizons} | {progress:5.1f}%]", flush=True)
            print(f"현재 시각       : {format_hhmm(hs)}  (t={hs} min)", flush=True)
            print(f"최적화 구간    : {format_hhmm(hs)} ~ {format_hhmm(he)}  ({he - hs}분)", flush=True)
            print(f"확정/Commit 구간: {format_hhmm(hs)} ~ {format_hhmm(commit_end)}  ({commit_end - hs}분)", flush=True)
            print(f"재계획 중첩구간 : {format_hhmm(overlap_start)} ~ {format_hhmm(overlap_end)}  ({max(0, overlap_end - overlap_start)}분)", flush=True)
            print(f"다음 갱신 시각 : {format_hhmm(commit_end)}", flush=True)
            print("-" * 72, flush=True)
            print(f"활성 Alternative Request : {len(active):,}", flush=True)
            print(f"활성 Batch               : {len(active_batch_ids):,}", flush=True)
            print(f"이번에 처음 활성화된 Batch: {len(new_batch_ids):,}", flush=True)
            print(f"이전 구간 Pending Batch   : {len(active_pending_batch_ids):,}", flush=True)
            print(f"현재 Onboard              : {onboard_batch_count:,} batch / {onboard_passenger_count:,} passenger", flush=True)
            print(f"누적 Complete / Overdue   : {complete_before:,} / {overdue_before:,}", flush=True)
            print("-" * 72, flush=True)

        objective, selected, committed, runtime = None, set(), 0, 0.0
        routes: list[dict[str, Any]] = []
        if active:
            if LOG_ROLLING_HORIZON:
                print("Solver 실행 시작...", flush=True)
            tick = time.perf_counter()
            """변경 시작: 중간 RH는 Open End, 마지막 처리 가능 RH는 최초 home_depot을 End로 설정"""
            return_to_home = he >= simulation_end
            model = build_horizon_model(active, batches, vehicles, distance, flight_time, hs, he, maximum_max_wait, return_to_home)
            """변경 끝"""
            solution = solve_horizon(model)
            runtime = time.perf_counter() - tick
            routes, selected = extract_routes(model, solution, vehicles)
            objective = None if solution is None else int(solution.ObjectiveValue())
            committed = commit_routes(routes, vehicles, distance, flight_time, nodes, commit_end, logs, batches)
            for task in active:
                """"""  # 변경 시작: planned_visit가 단순 계획인지 실제 Commit인지 구분할 수 있도록 planned Pickup 시각과 committed 여부를 plan history에 추가한다.
                planned_pickup_time = next((int(row["time"]) for row in routes if row["event"] == "Pickup" and row["task"].task_key == task.task_key), None)
                plans.append({"horizon_start": hs, "horizon_end": he, "request_id": task.request_id, "batch_id": task.batch_id, "planned_visit": task.task_key in selected, "planned_pickup_time": planned_pickup_time, "committed": planned_pickup_time is not None and planned_pickup_time < commit_end and task.status == "Complete", "drop_penalty": drop_penalty(batches[task.batch_id], maximum_max_wait)})
                """"""  # 변경 끝: committed=True는 해당 Pickup 계획시각이 현재 RH의 commit_end보다 앞서 실제 확정 대상이 된 경우를 의미한다.
        elif LOG_ROLLING_HORIZON:
            print("활성 Request가 없어 Solver 실행을 건너뜁니다.", flush=True)

        update_batch_states(batches, commit_end, increment_pending=True, attempted_batch_ids=active_batch_ids)
        complete_after = sum(b.status == "Complete" for b in batches.values())
        pending_after = sum(b.status == "Pending" for b in batches.values())
        overdue_after = sum(b.status == "Overdue" for b in batches.values())
        used_vehicle_count = len({int(row["vehicle_id"]) for row in routes})

        summaries.append({"horizon_start": hs, "horizon_end": he, "active_request_count": len(active), "planned_request_count": len(selected), "committed_request_count": committed, "completed_batch_count": complete_after, "pending_batch_count": pending_after, "overdue_batch_count": overdue_after, "objective_value": objective, "solver_runtime_seconds": round(runtime, 3)})

        if LOG_ROLLING_HORIZON:
            print("Solver 완료" if active else "구간 처리 완료", flush=True)
            print(f"Solver Runtime            : {runtime:.3f} sec", flush=True)
            print(f"Solution                  : {'FOUND' if objective is not None else ('SKIPPED' if not active else 'NOT FOUND')}", flush=True)
            print(f"Objective                 : {objective:,}" if objective is not None else "Objective                 : -", flush=True)
            print(f"계획된 Alternative Request: {len(selected):,}", flush=True)
            print(f"Commit된 Pickup           : {committed:,}", flush=True)
            print(f"계획에 사용된 Vehicle     : {used_vehicle_count:,}", flush=True)
            print(f"이번 구간 신규 Complete   : {complete_after - complete_before:,}", flush=True)
            print(f"이번 구간 신규 Overdue    : {overdue_after - overdue_before:,}", flush=True)
            print(f"현재 Pending Batch        : {pending_after:,}", flush=True)
            print(f"누적 Complete / Overdue   : {complete_after:,} / {overdue_after:,}", flush=True)
            if rh_index < total_horizons:
                print(f"다음 RH로 이동            : {format_hhmm(hs)} -> {format_hhmm(commit_end)} (+{REOPTIMIZATION_INTERVAL_MINUTES}분)", flush=True)
            else:
                print("마지막 Rolling Horizon 구간입니다.", flush=True)
            print("=" * 72, flush=True)

        seen_batch_ids.update(active_batch_ids)
    update_batch_states(batches, simulation_end + ROLLING_HORIZON_MINUTES)
    for batch in batches.values():
        if batch.status == "Pending":
            batch.status = "Overdue"
            for task in batch.alternatives:
                if task.status != "Complete":
                    task.status = "Overdue"
    if any(b.status == "Pending" for b in batches.values()):
        raise RuntimeError("최종 시뮬레이션에 Pending batch가 남았습니다")
    route_df = pd.DataFrame(logs)
    if not route_df.empty:
        route_df = route_df.sort_values(["vehicle_id", "departure_time", "arrival_time"]).reset_index(drop=True)
        route_df["sequence"] = route_df.groupby("vehicle_id").cumcount()
        """"""  # 변경 시작: vehicle_route_legs.csv의 분 단위 departure_time/arrival_time은 유지하고, BASE_TIME 기준 실제 시각 열을 추가한다.
        route_df["depT_hhmm"] = route_df["departure_time"].map(lambda value: format_hhmm(int(value)))
        route_df["arrT_hhmm"] = route_df["arrival_time"].map(lambda value: format_hhmm(int(value)))
        """"""  # 변경 끝: depT_hhmm/arrT_hhmm 열에 실제 출발·도착 시각(HH:MM)을 저장한다.

    """"""  # 변경 시작: 내부 route_df는 그대로 두고, 사람이 읽는 vehicle_route_legs.csv 전용 출력 복사본만 만든다. Solver/penalty/occupancy/vehicle summary에는 영향을 주지 않는다.
    route_output_df = route_df.copy()
    if not route_output_df.empty:
        same_node_stay = (
            route_output_df["event"].eq("Movement")
            & route_output_df["from_node"].astype(str).eq(route_output_df["to_node"].astype(str))
            & pd.to_numeric(route_output_df["travel_time"], errors="coerce").fillna(0).eq(0)
            & pd.to_numeric(route_output_df["travel_distance_km"], errors="coerce").fillna(0).eq(0)
        )

        # 0거리 Movement는 실제 비행이 아니므로 CSV 표시에서만 Stay로 구분한다.
        route_output_df.loc[same_node_stay, "event"] = "Stay"
        route_output_df.loc[same_node_stay, "movement_type"] = "parking_charging"

        # 모든 event를 start -> end 방향으로 통일한다.
        # Movement: 실제 출발 -> 실제 도착
        # Pickup/Delivery: 서비스 시작(도착) -> 서비스 종료(3분 후)
        # Stay: 대기/충전 시작 -> 다음 event 시작 직전
        movement_mask = route_output_df["event"].eq("Movement")
        service_mask = route_output_df["event"].isin(["Pickup", "Delivery"])
        stay_mask = route_output_df["event"].eq("Stay")

        route_output_df["start_time"] = route_output_df["arrival_time"]
        route_output_df["end_time"] = route_output_df["departure_time"]

        route_output_df.loc[movement_mask, "start_time"] = route_output_df.loc[movement_mask, "departure_time"]
        route_output_df.loc[movement_mask, "end_time"] = route_output_df.loc[movement_mask, "arrival_time"]

        stay_start = (
            pd.to_numeric(route_output_df.loc[stay_mask, "departure_time"], errors="raise")
            - pd.to_numeric(route_output_df.loc[stay_mask, "charging_time"], errors="raise")
        ).clip(lower=0)
        route_output_df.loc[stay_mask, "start_time"] = stay_start
        route_output_df.loc[stay_mask, "end_time"] = route_output_df.loc[stay_mask, "departure_time"]

        route_output_df["start_time"] = pd.to_numeric(route_output_df["start_time"], errors="raise").astype(int)
        route_output_df["end_time"] = pd.to_numeric(route_output_df["end_time"], errors="raise").astype(int)
        route_output_df["duration_min"] = route_output_df["end_time"] - route_output_df["start_time"]
        route_output_df["start_time_hhmm"] = route_output_df["start_time"].map(format_hhmm)
        route_output_df["end_time_hhmm"] = route_output_df["end_time"].map(format_hhmm)

        # Pickup/Delivery의 duration_min은 SERVICE_TIME_MINUTES=3이다.
        # charging_time=BOARDING_CHARGE_MINUTES=2는 이 3분 안에서 동시에 충전되는 시간이므로
        # 서비스 총 소요시간을 5분으로 늘리지 않는다.
        """"""  # 변경 시작: vehicle_route_legs.csv에서는 start_time/end_time으로 시간 표현을 통일하므로 departure_time/arrival_time/depT_hhmm/arrT_hhmm은 출력에서 제외한다. 내부 route_df의 departure_time/arrival_time 계산은 그대로 유지한다.
        route_output_df = route_output_df[
            [
                "vehicle_id",
                "sequence",
                "event",
                "from_node",
                "to_node",
                "physical_from_node",
                "physical_to_node",
                "start_time",
                "end_time",
                "start_time_hhmm",
                "end_time_hhmm",
                "duration_min",
                "travel_time",
                "travel_distance_km",
                "charging_time",
                "charged_range_km",
                "remaining_range_before",
                "remaining_range_after",
                "onboard_passengers",
                "related_batch_id",
                "movement_type",
                "from_event",
                "to_event",
            ]
        ]
        """"""  # 변경 끝: 사람이 확인하는 vehicle_route_legs.csv에는 start_time/end_time 계열만 남기며, 기존 내부 시간 계산 및 다른 출력 파일에는 영향을 주지 않는다.
    """"""  # 변경 끝: vehicle_route_legs.csv에서 Movement/Service/Stay의 시간을 모두 start_time -> end_time 순서로 읽을 수 있게 했으며, 계산 로직은 변경하지 않았다.

    save_csv(batch_dataframe(batches, maximum_max_wait), "rolling_horizon_request_status.csv")
    save_csv(route_output_df, "vehicle_route_legs.csv")
    save_csv(pd.DataFrame(plans), "rolling_horizon_plan_history.csv")
    save_csv(pd.DataFrame(summaries), "rolling_horizon_summary.csv")
    # [추가 수정 4] 물리적 차량 위치 복원을 위해 초기 home_depot 정보가 있는 vehicles를 함께 전달한다.
    save_csv(occupancy_dataframe(route_df, nodes, vehicles), "node_vehicle_occupancy_by_minute.csv")
    # [추가 수정 4 끝] 출력 파일명과 다른 출력 로직은 그대로 유지한다.
    save_csv(vehicle_summary_dataframe(route_df, vehicles), "vehicle_summary.csv")
    print(f"Complete={sum(b.status == 'Complete' for b in batches.values()):,}, Overdue={sum(b.status == 'Overdue' for b in batches.values()):,}")
    print(f"runtime={time.perf_counter() - started:.2f}s, output={OUTPUT_DIR}")


if __name__ == "__main__":
    main()

