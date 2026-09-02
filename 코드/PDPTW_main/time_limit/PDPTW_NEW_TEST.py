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


PROJECT_DIR = Path(__file__).resolve().parents[3]
INPUT_DIR = PROJECT_DIR / "자료" / "기초자료"
REQUEST_PATH = INPUT_DIR / "finalDemand_v5" / "finalDemand_v5" / "d5000_s01.csv"
DISTANCE_MATRIX_PATH = INPUT_DIR / "distance_matrix_km.csv"
TIME_MATRIX_PATH = INPUT_DIR / "flight_time_matrix_min_remove_fuel.csv"
NODE_REFERENCE_PATH = INPUT_DIR / "vp_reference.csv"
TRANSPORTATION_MATRIX_PATH = PROJECT_DIR / "자료" / "결과" / "차량_교통수단" /"public_transit_time_matrix_tmap_min.csv"
TRANSPORTATION_MATRIX_COST = PROJECT_DIR / "자료" / "결과" / "차량_교통수단" / "public_transit_fare_matrix_tmap_krw.csv"
OUTPUT_DIR = PROJECT_DIR / "자료" / "결과" / "Ortools" / "d4000" / "Penalty_Per_Vehicles" / "Original"

BASE_TIME = "05:40"
END_TIME = "19:16"
NUM_VEHICLES = 176
VEHICLE_CAPACITY = 4
DEPOT_ROUTE_NODE_IDS = list(range(1, 11))
ROLLING_HORIZON_MINUTES = 60
REOPTIMIZATION_INTERVAL_MINUTES = 20
TIME_LIMIT_SECONDS = 20

SERVICE_TIME_MINUTES = 3
BOARDING_CHARGE_MINUTES = 2
TAXI_TIME_MINUTES = 1
REVENUE = 100000
HOVERING_LIFT_OFF_COST = 18404
MIN_PER_OPERATING = 1247
MIN_PER_MECHANIC = 9140
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

# load_task(), load_nodes()와 같이 csv를 통해 불러올 때, 사용
def find_column(df: pd.DataFrame, key: str, required: bool = True) -> str | None:
    names = {str(c).strip().lower(): str(c) for c in df.columns}
    for alias in COLUMN_ALIASES[key]:
        if alias.lower() in names:
            return names[alias.lower()]
    if required:
        raise KeyError(f"필수 열 '{key}'이 없습니다. 허용 alias={COLUMN_ALIASES[key]}, 실제 열={list(df.columns)}")
    return None

# 인코딩, 해당 파일을 열기 전에 여러가지 방식으로 인코딩을 진행.
def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"CSV encoding을 판별할 수 없습니다: {path}")

# Distance & Time matrix를 불러옴
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
    rc = find_column(df, "route_node") # df에서 "route_node"를 가져옴. 정확히는 COLUMN_ALIASES의 route_node list에서 "vp_id"를 가져옴.
    pc = find_column(df, "physical_node", False) or rc # 이전에는 physical_node라고 지정을 했지만, 이제는 없어서 FALSE로 설정
    ac = find_column(df, "address", False) # 같음
    xc, yc = find_column(df, "longitude"), find_column(df, "latitude") # lat, lon으로 불러옴
    return {normalize_id(r[rc]): NodeInfo(normalize_id(r[rc]), normalize_id(r[pc]), "" if ac is None else str(r[ac]), float(r[xc]), float(r[yc])) for _, r in df.iterrows()}

# vp_reference.csv에 대한 정보로 각 A->B의 case를 통해 모든 경우의 수를 matrix 형태로 생성
def load_named_matrix(path: Path, nodes: dict[str, NodeInfo]) -> pd.DataFrame:
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

# 모든 비행시간에 대한 최종 종료 시간. 단, 본 코드는 착륙까지 end_time내에 하는 것이 아닌 단지 최종 경로 선택 (commit)을 시점으로 진행한다.
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

def Value_Of_Time(
    batch: BatchState,
    fare_matrix: pd.DataFrame,
    revenue: int
) -> float:

    representative = batch.alternatives[0]

    public_transit_fare = float(
        fare_matrix.loc[
            representative.origin,
            representative.destination
        ]
    )

    time_saving = (
        representative.transportation_min
        - representative.joby_min
    )

    if time_saving <= 0:
        return 0.0

    vot = (
        revenue - public_transit_fare
    ) / time_saving

    return vot


def Pending_Cost(batch: BatchState, fare_matrix: pd.DataFrame, revenue: int, hs):
    representative = batch.alternatives[0]
    Number_Of_Pending = batch.pending_count
    passengers = representative.passengers
    vot = Value_Of_Time(batch, fare_matrix, revenue)
    remaining_time = representative.window_end - hs
    alpha = 0

    if remaining_time <= 5:
        alpha = 1
    elif 5 < remaining_time <= 10:
        alpha = 0.75
    elif 10 < remaining_time <= 15:
        alpha = 0.5
    else:
        alpha = 0.25
 
    return int(vot * Number_Of_Pending * REOPTIMIZATION_INTERVAL_MINUTES * passengers * alpha)

def Drop_Cost(batch: BatchState, revenue:int):
    representative = batch.alternatives[0]
    passengers = representative.passengers
    cost = revenue

    return int(passengers * cost)

def Tranportation_Joby_fare(batch: BatchState, fare_matrix: pd.DataFrame, revenue: int):
    representative = batch.alternatives[0]
    transportation_time = representative.transportation_min
    joby_min = representative.joby_min
    Vot = Value_Of_Time(batch, fare_matrix, revenue)
    passengers = representative.passengers

    return int(Vot * (transportation_time - joby_min) * passengers)


def total_adddisjunction_cost(batch: BatchState, fare_matrix: pd.DataFrame, revenue: int, hs):
    Pending_Cost_1 = Pending_Cost(batch, fare_matrix, revenue, hs)
    Drop_Cost_1 = Drop_Cost(batch, revenue)
    Tranportation_Joby_fare_1 = Tranportation_Joby_fare(batch, fare_matrix, revenue)

    return int(Pending_Cost_1 + Drop_Cost_1 + Tranportation_Joby_fare_1)


""" main에서 생성한 DataFrame을 가져옴. 최종적으로 Request OD에 대한 batch를 생성하기 위함. 왜냐하면, Request_OD가 1->2 일 때 passenger가 4명이면 [3,1]로 분리를 해야하므로 이때 사용하는 게 Batch"""
def load_tasks(request_path: Path, distance: pd.DataFrame, flight_time: pd.DataFrame, nodes: dict[str, NodeInfo], transportation_matrix: pd.DataFrame | None = None) -> tuple[list[RequestTask], dict[str, BatchState]]:
    raw = read_table(request_path) # d5000.csv를 가져옴
    oc, dc = find_column(raw, "origin"), find_column(raw, "destination") # origin = n, destination = m
    cc, wc = find_column(raw, "passengers"), find_column(raw, "max_wait")  # passengers = cnt, max_wait = maxWait_min
    depc = find_column(raw, "departure", False)
    arrc = find_column(raw, "arrival", False)
    tc, jc = find_column(raw, "transportation", False), find_column(raw, "joby_time", False)
    ric, bic = find_column(raw, "request_id", False), find_column(raw, "batch_id", False)
    tasks: list[RequestTask] = []
    for pos, (_, row) in enumerate(raw.iterrows()):
        origin, destination = normalize_id(row[oc]), normalize_id(row[dc]) # 각 출발지와 도착지의 모든 data를 불러옴.
        if origin == destination: # n과 m이 같을 경우 생성 x
            continue 
        if origin not in nodes or destination not in nodes or origin not in distance.index or destination not in distance.columns:
            raise KeyError(f"수요 row {pos}의 node가 행렬/참조에 없습니다: {origin}->{destination}")
        if depc:
            art = to_offset_minutes(row[depc])
        # Ortools 시간으로 변경    
        elif {"depT_hr", "depT_m"}.issubset(raw.columns):
            art = int(row["depT_hr"]) * 60 + int(row["depT_m"]) - base_time_minutes()
        else:
            raise KeyError("필수 arrival time 열(art/departure 또는 depT_hr+depT_m)이 없습니다")
        departure_time = max(0, art) # Base_time을 5:40으로 설정했으나, 아주 극악의 확률로 5:30분을 반영하게 되는 경우, art < 0이 됨. ortools는 음수는 허용하지 않으므로 그냥 0으로 출력하는 안전장치
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
        request_id = normalize_id(row[ric]) if ric else f"R{pos + 1}" # 위에 있는 ric는 전부 none이다. 따라서, else의 R{pos + 1}을 통해서 orgin과 destination의 열의 개수만큼 Node를 생성
        batch_id = normalize_id(row[bic]) if bic else f"B{pos + 1}" # 위와 동일
        count = int(math.ceil(float(row[cc]))) 
        if count < 1:
            raise ValueError(f"row {pos}: cnt는 1 이상이어야 합니다: {count}")
        if bic and count > VEHICLE_CAPACITY:
            raise ValueError(f"row {pos}: 명시적 batch_id의 cnt={count}가 vehicle capacity={VEHICLE_CAPACITY}를 초과합니다")
        # Request_OD가 차량의 Vehicles보다 많은 경우
        parts = int(math.ceil(count / VEHICLE_CAPACITY))
        remaining = count
        # Passenger > 차량 capacity임을 위해 batch_id를 생성
        for part in range(parts):
            passengers = min(VEHICLE_CAPACITY, remaining)
            part_batch = batch_id if parts == 1 else f"{batch_id}-{part + 1}"
            part_request = request_id if parts == 1 else f"{request_id}-{part + 1}"
            tasks.append(RequestTask(f"{part_batch}:{part_request}", part_request, part_batch, origin, destination, passengers, max(0, art), departure_time, arrival_time, int(math.ceil(float(row[wc]))), transportation, joby, float(distance.loc[origin, destination]))) # task 내에 전처리한 정보를 저장
            remaining -= passengers
    batches: dict[str, BatchState] = {} #위의 BatchState 클래스 양식에 맞게 생성
    for task in tasks:
        batches.setdefault(task.batch_id, BatchState(task.batch_id)).alternatives.append(task)
    return tasks, batches # 궁극적으로 dict 형태로 출력



def build_horizon_model(active: list[RequestTask], batches: dict[str, BatchState], vehicles: list[VehicleState], distance: pd.DataFrame, flight_time: pd.DataFrame, fare_matrix: pd.DataFrame, hs: int, he: int, maximum_max_wait: int, return_to_home: bool = False) -> HorizonModel:
    vehicle_count = len(vehicles)
    starts = list(range(vehicle_count)) # starts 와 ends는 궁극적으로 depot이므로 각 차량 수 만큼 가상의 depot node를 생성해야한다. 단, 아직 실제 node로 변환 된 것은 아님. def location(model_node:int)를 참조
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
            return by_vehicle[model_node].route_node_id # 위의 vehicles는 class VehicleState에서 가져온 것으로 가상의  node -> physical node로 변환(start)
        # 차량의 원래 Depot
        if model_node < vehicle_count * 2: 
            return by_vehicle[model_node - vehicle_count].home_depot # 위의 vehicles는 class VehicleState에서 가져온 것으로 가상의  node -> physical node로 변환(start)
        # Request Node
        event, task = node_meta[model_node]
        return task.origin if event == "Pickup" else task.destination  # type: ignore[union-attr]

    def service(model_node: int) -> int:
        return SERVICE_TIME_MINUTES if model_node in node_meta else 0 #어차피 모든 노드는 전부 PD 노드 그래서 Delivery 이후에 SERVICE_TIME_MINUTES를 진행.

    def time_cb(fi: int, ti: int) -> int:
        f, t = manager.IndexToNode(fi), manager.IndexToNode(ti)
        if not return_to_home and vehicle_count <= t < vehicle_count * 2: # 각 RH 당 형식상 depot이 필요하다. 따라서, 위의 가상의 노드를 가져와 (마지막_노드) -> (가상_Depot)을 0으로 설정
            return service(f) #이를 통해, (마지막_노드)에 대한 최종시간 출력. 왜냐하면, def service()를 통해서 (마지막_노드) -> (가상_Depot)에 대한 값은 0임
        return service(f) + int(flight_time.loc[location(f), location(t)])

    def distance_cb(fi: int, ti: int) -> int:
        f, t = manager.IndexToNode(fi), manager.IndexToNode(ti)
        if not return_to_home and vehicle_count <= t < vehicle_count * 2: # 각 RH 당 형식상 depot이 필요하다. 따라서, 가상의 depot을 생성해 (마지막_노드) -> (가상_Depot)을 0으로 설정
            return 0
        return int(round(float(distance.loc[location(f), location(t)]) * 1000))

    def cost_cb(fi: int, ti:int) -> int:
        f, t = manager.IndexToNode(fi), manager.IndexToNode(ti)

        if not return_to_home and vehicle_count <= t <vehicle_count * 2:
            return 0
        flight_min = int(flight_time.loc[location(f), location(t)])
        return HOVERING_LIFT_OFF_COST + flight_min * (MIN_PER_OPERATING + MIN_PER_MECHANIC)
    
    def range_cb(fi: int, ti: int) -> int: # 각 경로별 거리 출력 이 거리를 통해 vehicle의 잔여 비행가능 거리를 업데이트 
        return -distance_cb(fi, ti) # 각 경로별 거리 출력 이 거리를 통해 vehicle의 잔여 비행가능 거리를 업데이트 

    ti = routing.RegisterTransitCallback(time_cb) # 누적 비행 시간
    di = routing.RegisterTransitCallback(distance_cb) # 경로별 거리
    ri = routing.RegisterTransitCallback(range_cb) # 비행기 잔여 비행가능 거리 
    ci = routing.RegisterTransitCallback(cost_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(ci) # distance에 대한 cost 평가
    model_end = he + int(flight_time.to_numpy().max()) * 4 + SERVICE_TIME_MINUTES * 4 # 비행 마무리 시간에 대한 여유 분 제공. -> 일몰 시간(End_Time) 이전의 request_OD를 전부 처리하고 Depot으로 복귀하는 여유시간 
    routing.AddDimension(ti, model_end, model_end, False, "Time") # 선택한 모든 경로들 model_end 이전에 끝내도록 설정
    td = routing.GetDimensionOrDie("Time")
    max_range_m, reserve_m = int(MAX_REMAINING_RANGE_KM * 1000), int(MIN_REMAINING_RANGE_KM * 1000) # 최대 비행가능 거리 및 최소 비행가능 거리에 대한 설정
    routing.AddDimension(ri, max_range_m, max_range_m, False, "RemainingRange") # 각 process마다 계산을 진행
    rd = routing.GetDimensionOrDie("RemainingRange") # 위의 각 process를 진행할 때마다 값을 축적해서 계산을 진행, 즉, remaining_range를 업데이트
    solver = routing.solver()
    charge_rate_m = int(round(CHARGING_RATE_KM_PER_MIN * 1000))

    for v in vehicles:
        si, ei = routing.Start(v.vehicle_id), routing.End(v.vehicle_id)
        start_time = max(hs, v.available_time)
        td.CumulVar(si).SetValue(start_time) # 각 기체 시작 시간(모두, 5:40에 시작)
        td.CumulVar(ei).SetRange(start_time, model_end) # 각 기체 시작 및 종료 시간(모두 5:40~19:16으로 설정)
        rd.CumulVar(si).SetValue(int(round(v.remaining_range_km * 1000))) # 잔여비행거리 설정(모두 시작은 만땅인 160으로 지정)
        rd.CumulVar(ei).SetRange(reserve_m, max_range_m) # 최소 및 최대 거리(모두 종료는 (최소, 최대) 사이의 범위에 있게 함)
        routing.AddVariableMinimizedByFinalizer(td.CumulVar(ei)) # solver를 통해 생성된 여러의 rolling_Horizon의 end의 결과값에서 제일 작은 값을 출력

    by_batch: dict[str, list[int]] = {}
    """각 Request_Node의 모든 column을 전처리 작업(Time_Window, pickup & delivery, 비행 조건 등)을 진행"""
    for task in active:
        pnode, dnode = pickup_nodes[task.task_key], delivery_nodes[task.task_key]
        pi, deli = manager.NodeToIndex(pnode), manager.NodeToIndex(dnode)
        td.CumulVar(pi).SetRange(max(hs, task.departure_time), he) # 사실상 [이륙 시간, Rolling_Horizon의 end]
        td.CumulVar(deli).SetRange(max(hs, task.arrival_time), min(task.window_end, end_time_minutes())) # task.window_end에서 arrival_time + max_Wait()가 함유
        routing.AddPickupAndDelivery(pi, deli)
        solver.Add(routing.VehicleVar(pi) == routing.VehicleVar(deli)) # pickup과 Delivery를 같은 차량이 수행
        solver.Add(td.CumulVar(pi) <= td.CumulVar(deli)) # Pickup과 delivery 이므로 누적시간이 pickup보다 delivery가 더 커야함
        solver.Add(routing.ActiveVar(pi) == routing.ActiveVar(deli)) # pickup을 실행했다면, 반드시 delivery도 실행.
        routing.AddDisjunction([deli], 0) # delivery(drop)에 대한 penalty를 0으로 한다.
        by_batch.setdefault(task.batch_id, []).append(pi) # 위에서 설정한 batch_id를 불러옴
        for idx in (pi, deli):
            rd.CumulVar(idx).SetRange(reserve_m, max_range_m)
            rd.SlackVar(idx).SetRange(0, max_range_m) # 대기 시간동안 허용되는 배터리 충전 범위 / 하지만, 밑에 추가 조건이 존재
            solver.Add(rd.SlackVar(idx) <= charge_rate_m * (td.SlackVar(idx) + BOARDING_CHARGE_MINUTES)) # 대기시간동안 충전되는 양 <= 4.267 * (대기시간 + 승하차_시간동안의 충전) 
            solver.Add(rd.CumulVar(idx) + rd.SlackVar(idx) <= max_range_m) # 기존 잔량 + 대기 시간동안의 충전된 양이 max_range이전까지여야 함.

    for batch_id, pickup_indices in by_batch.items():
        penalty = total_adddisjunction_cost(batches[batch_id], fare_matrix, REVENUE, hs)
        routing.AddDisjunction(pickup_indices, penalty, 1) # 실제 반영은 이곳에서 진행

    demands = [0] * cursor
    for node, (event, task) in node_meta.items():
        demands[node] = task.passengers if event == "Pickup" else -task.passengers  # type: ignore[union-attr]
    demand_idx = routing.RegisterUnaryTransitCallback(lambda index: demands[manager.IndexToNode(index)])
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [VEHICLE_CAPACITY] * vehicle_count, True, "Capacity")
    return HorizonModel(manager, routing, td, rd, node_meta, pickup_nodes, delivery_nodes) 

# 여기서 계산을 진행
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
    state_by_id = {v.vehicle_id: v for v in vehicles} # 각 자동차_id에 기록된 state를 부여 
    committed = 0 # pickup이 확정된 경우를 committed라고 한다. 이때, committed가 되지 않은 경우 Pending 혹은 Overdue가 됨. 주된 목적은 RH가 겹치므로 겹치는 구간에 대해 먼저 경로 계산 및 기록 이후에 겹친 구간에 대해 Pending 등의 Penalty를 부여
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in route_events:
        grouped.setdefault(row["vehicle_id"], []).append(row)
    for vehicle_id, events in grouped.items():
        state = state_by_id[vehicle_id]
        chosen = {r["task"].task_key for r in events if r["event"] == "Pickup" and r["time"] < commit_end} # commit_end는 Reoptimization의 시작 부근까지 진행. 겹치는 구간은 계산만 진행 -> Pending(우선권)을 부여.
        if not chosen: # 즉, Pickup이 commit 이전까지 진행된 것들만 추려둔다.
            continue

        projected_node, projected_time, projected_remaining = state.route_node_id, state.available_time, state.remaining_range_km 
        late_task_keys: set[str] = set() # END_TIME보다 늦게 도착한 Delivery에 대한 list 추가. 이들은 그 다음 RH에서 Pending을 통해 우선권이 부여되거나 Overdue가 될 것이다.
        """해당 for문의 경우는 Delivery를 완료하기 까지 진행하는데, 필요한 것들이다. 즉, 위에서 간추려진 pickup list에서 PD를 진행할 때, 아래의 조건(충전, 대기 등)을 고려해야하고 이를 통해 Delivery가 end_time 이내에 진행되는지를 분석. 이내에 못했다면, commit 실패 pending이나 overdue로 등록"""
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
        chosen -= late_task_keys # pickup에서는 가능했지만 Delivery에서 실패한 구간을 제외
        if not chosen:
            continue
        # Commit the complete pickup/delivery sequence for pickups inside the frozen interval.
        events = [r for r in events if r["task"].task_key in chosen] # 최종적으로 Commit하기로 남은 chosen Request들의 Pickup/Delivery 이벤트만 추려냄.
        onboard = dict(state.onboard_batches) #차량에 타고 있는 Batch 별 승객 정보
        """onboard = {
                    "B10": 3,
                    "B21": 1
            }의 형태. max_capacity가 4이므로 최대 onboard는 4여야 한다."""
        current_node, current_time, remaining = (state.route_node_id, 
                                                 state.available_time, 
                                                 state.remaining_range_km)
        for row in events:
            task, event = row["task"], row["event"]
            target = task.origin if event == "Pickup" else task.destination
            dist, travel = float(distance.loc[current_node, target]), int(flight_time.loc[current_node, target])
            planned = max(current_time, row["time"])
            parking = max(0, planned - current_time - travel) # 해당 노드에서 움직이지 않고 대기하는 시간
            range_before_stay = remaining
            remaining = min(MAX_REMAINING_RANGE_KM, remaining + parking * CHARGING_RATE_KM_PER_MIN)
            extra_charge, departure_range = charge_for_departure(remaining, dist)
            departure = current_time + parking + extra_charge # 실제 이륙시간
            arrival = departure + travel # 실제 도착한 시간
            after_flight = departure_range - dist # 잔여항속거리
            before_count = sum(onboard.values()) #현재 탑승한 승객 수
            related = task.batch_id

            if parking + extra_charge > 0: # 해당 상태는 stay를 의미. 따라서, 이때, stay라는 조건을 기록
                log_rows.append({"vehicle_id": vehicle_id, "sequence": len(log_rows), "event": "Stay", "from_node": current_node, "to_node": current_node, "physical_from_node": nodes[current_node].physical_node_id, "physical_to_node": nodes[current_node].physical_node_id, "departure_time": departure, "arrival_time": current_time, "travel_time": 0, "travel_distance_km": 0.0, "charging_time": parking + extra_charge, "charged_range_km": round(departure_range - range_before_stay, 3), "remaining_range_before": round(range_before_stay, 3), "remaining_range_after": round(departure_range, 3), "onboard_passengers": before_count, "related_batch_id": related, "movement_type": "parking_charging", "from_event": "Position", "to_event": "Movement" if current_node != target else event})

            if current_node != target: # Node에 있지 않을 때는 Movement로 이동
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

# completed, overdue의 여부에 따라 Node 상태를 업데이트
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
            batch.status = "Pending"
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
        if charge_min > 0:
            logs.append({"vehicle_id": state.vehicle_id, "sequence": len(logs), "event": "Stay", "from_node": state.route_node_id, "to_node": state.route_node_id, "physical_from_node": nodes[state.route_node_id].physical_node_id, "physical_to_node": nodes[state.route_node_id].physical_node_id, "departure_time": departure, "arrival_time": state.available_time, "travel_time": 0, "travel_distance_km": 0.0, "charging_time": charge_min, "charged_range_km": round(departure_range - state.remaining_range_km, 3), "remaining_range_before": round(state.remaining_range_km, 3), "remaining_range_after": round(departure_range, 3), "onboard_passengers": 0, "related_batch_id": "", "movement_type": "parking_charging", "from_event": "Position", "to_event": "Movement"})
        logs.append({"vehicle_id": state.vehicle_id, "sequence": len(logs), "event": "Movement", "from_node": state.route_node_id, "to_node": state.home_depot, "physical_from_node": nodes[state.route_node_id].physical_node_id, "physical_to_node": nodes[state.home_depot].physical_node_id, "departure_time": departure, "arrival_time": arrival, "travel_time": arrival - departure, "travel_distance_km": round(dist, 3), "charging_time": 0, "charged_range_km": 0.0, "remaining_range_before": round(departure_range, 3), "remaining_range_after": round(departure_range - dist, 3), "onboard_passengers": 0, "related_batch_id": "", "movement_type": "return_to_depot", "from_event": "Position", "to_event": "Depot"})
        state.route_node_id, state.available_time, state.node_arrival_time, state.remaining_range_km = state.home_depot, arrival, arrival, departure_range - dist


def format_hhmm(value: int | None) -> str:
    if value is None:
        return ""
    absolute = base_time_minutes() + int(value)
    return f"{(absolute // 60) % 24:02d}:{absolute % 60:02d}"


def batch_dataframe(batches: dict[str, BatchState], hs, fare_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for batch in batches.values():
        selected = next((t for t in batch.alternatives if t.request_id == batch.selected_request_id), batch.alternatives[0])
        # [추가 수정 1] rolling_horizon_request_status.csv에 각 Request의 원래 Delivery Time Window를 분 단위와 HH:MM 형식으로 함께 출력한다.
        rows.append({"batch_id": batch.batch_id, "selected_request_id": batch.selected_request_id, "origin": selected.origin, "destination": selected.destination, "max_wait": selected.max_wait_min, "time_window_start": selected.arrival_time, "time_window_start_hhmm": format_hhmm(selected.arrival_time), "time_window_end": selected.window_end, "time_window_end_hhmm": format_hhmm(selected.window_end), "Transportation_min": selected.transportation_min, "Joby_min": selected.joby_min, "Transportation_Joby_time_saving": selected.transportation_min - selected.joby_min, "pending_count": batch.pending_count, "final_cost(원)": total_adddisjunction_cost(batch, fare_matrix, REVENUE, hs), "assigned_vehicle": batch.assigned_vehicle_id, "pickup_time": batch.pickup_time, "pickup_time_hhmm": format_hhmm(batch.pickup_time), "delivery_time": batch.delivery_time, "delivery_time_hhmm": format_hhmm(batch.delivery_time), "final_status": batch.status})
        # [추가 수정 1 끝] time_window_start=arrival_time, time_window_end=arrival_time+max_wait이며 기존 계산/상태/패널티 로직은 변경하지 않는다.
    return pd.DataFrame(rows)


def occupancy_dataframe(route_df: pd.DataFrame, nodes: dict[str, NodeInfo], vehicles: list[VehicleState]) -> pd.DataFrame:
    """Preserve the minute-level vertiport occupancy output using each vehicle's physical location."""
    # 각 Vehicle을 t=0의 home_depot에서 시작시키고 Movement 구간만 비행 중으로 제외하여,
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

            # [departure, arrival] 동안은 비행 중이므로 어떤 노드에도 포함하지 않는다.
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
    # 기존 CSV 열 구조는 유지하고 vehicle_count의 의미만 '서비스 중 차량 수'에서
    # '해당 시각에 해당 노드에 실제로 존재하는 차량 수'로 변경한다.
    return result[columns]


def vehicle_summary_dataframe(route_df: pd.DataFrame, vehicles: list[VehicleState], batches: dict[str, BatchState], hs, fare_matrix, task) -> pd.DataFrame:
    rows = []
    for state in vehicles:
        total_passengers = sum(
            next(
                (
                task.passengers
                for task in batch.alternatives
                if task.request_id == batch.selected_request_id
            ),
            0
        )
        for batch in batches.values()
        if batch.assigned_vehicle_id == state.vehicle_id
        and batch.status == "Complete"
        )
        vr = route_df[route_df["vehicle_id"] == state.vehicle_id] if not route_df.empty else pd.DataFrame()
        stay = (
            vr[vr["event"] == "Stay"]
            if not vr.empty
            else pd.DataFrame()
        )

        service = (
            vr[vr["event"].isin(["Pickup", "Delivery"])]
            if not vr.empty
            else pd.DataFrame()
        )
        
        movement = vr[vr["event"] == "Movement"] if not vr.empty else pd.DataFrame()
        passenger = movement[movement["movement_type"] == "passenger_flight"] if not movement.empty else pd.DataFrame()
        empty = movement[movement["movement_type"].isin(["empty_repositioning", "return_to_depot"])] if not movement.empty else pd.DataFrame()
        stay_time = (
            int((stay["departure_time"] - stay["arrival_time"]).sum())
            if not stay.empty
            else 0
        )

        service_time = (
            int((service["departure_time"] - service["arrival_time"]).sum())
            if not service.empty
            else 0
        )

        ground_activity_time = stay_time + service_time
        total_flight_time = (
            int(movement["travel_time"].sum())
            if not movement.empty
            else 0
        )

        # 얘는 순수 운영비용.
        operating_cost_per_vehicle = int(
            len(movement) * HOVERING_LIFT_OFF_COST
            + total_flight_time
            * (MIN_PER_OPERATING + MIN_PER_MECHANIC)
        )
        rows.append(
                    {"vehicle_id": state.vehicle_id, 
                     "home_depot": state.home_depot, 
                     "final_node": state.route_node_id, 
                     "final_available_time": state.available_time, 
                     "Total_Passenger":total_passengers, 
                     "Total_Movement": int(len(movement)) if not movement.empty else 0, 
                     "Passenger_Movement": int(len(passenger)) if not passenger.empty else 0, 
                     "Empty_Movement": int(len(empty)),
                     "total_flight_time_min": int(movement["travel_time"].sum()) if not movement.empty else 0,
                     "total_ground_time_min": int(ground_activity_time), 
                     "total_flight_distance_km": round(float(movement["travel_distance_km"].sum()), 3) if not movement.empty else 0.0, 
                     "passenger_flight_distance_km": round(float(passenger["travel_distance_km"].sum()), 3) if not passenger.empty else 0.0, 
                     "empty_flight_distance_km": round(float(empty["travel_distance_km"].sum()), 3) if not empty.empty else 0.0, 
                     "charging_time_min": int(vr["charging_time"].sum()) if not vr.empty else 0, 
                     "Total_Cost": int(operating_cost_per_vehicle)}
                     )
        
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")


def main() -> None:
    started = time.perf_counter()
    distance, flight_time, nodes = load_matrix(DISTANCE_MATRIX_PATH, False), load_matrix(TIME_MATRIX_PATH, True), load_nodes(NODE_REFERENCE_PATH)
    transportation_matrix = load_named_matrix(TRANSPORTATION_MATRIX_PATH, nodes)
    fare_matrix = load_named_matrix(TRANSPORTATION_MATRIX_COST, nodes)
    if set(distance.index) != set(flight_time.index) or not set(distance.index).issubset(nodes):
        raise ValueError("distance/time/node-reference의 node 집합이 일치하지 않습니다")
    depots = [normalize_id(v) for v in DEPOT_ROUTE_NODE_IDS]
    missing = [v for v in depots if v not in distance.index]
    if missing:
        raise ValueError(f"Depot node가 행렬에 없습니다: {missing}")
    tasks, batches = load_tasks(REQUEST_PATH, distance, flight_time, nodes, transportation_matrix)
    maximum_max_wait = max(t.max_wait_min for t in tasks)
    latest_window_end = max(t.window_end for t in tasks)
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
        active = [t for t in tasks if batches[t.batch_id].status == "Pending" and t.art <= he and t.window_end >= hs and t.arrival_time <= simulation_end]
        active_batch_ids = {t.batch_id for t in active}
        new_batch_ids = active_batch_ids - seen_batch_ids
        active_pending_batch_ids = {batch_id for batch_id in active_batch_ids if batches[batch_id].pending_count > 0}
        onboard_batch_count = sum(len(v.onboard_batches) for v in vehicles)
        onboard_passenger_count = sum(sum(v.onboard_batches.values()) for v in vehicles)
        complete_before = sum(b.status == "Complete" for b in batches.values())
        overdue_before = sum(b.status == "Overdue" for b in batches.values())
        total_passengers = sum(task.passengers for task in tasks)

        total_batches = len(batches)

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
        demand_summary_df = pd.DataFrame([{
            "base_time": BASE_TIME,
            "end_time": END_TIME,
            "total_passengers": total_passengers,
            "total_batches": total_batches,
            "completed_passengers": completed_passengers,
            "overdue_passengers": overdue_passengers,
            "service_rate_percent":
                completed_passengers / total_passengers * 100
                if total_passengers > 0 else 0,
        }])

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
            model = build_horizon_model(active, batches, vehicles, distance, flight_time, fare_matrix, hs, he, maximum_max_wait, return_to_home)
            """변경 끝"""
            solution = solve_horizon(model)
            runtime = time.perf_counter() - tick
            routes, selected = extract_routes(model, solution, vehicles)
            objective = None if solution is None else int(solution.ObjectiveValue())
            committed = commit_routes(routes, vehicles, distance, flight_time, nodes, commit_end, logs, batches)
            for task in active:
                planned_pickup_time = next((int(row["time"]) for row in routes if row["event"] == "Pickup" and row["task"].task_key == task.task_key), None)
                plans.append({"horizon_start": hs, "horizon_end": he, "request_id": task.request_id, "batch_id": task.batch_id, "planned_visit": task.task_key in selected, "planned_pickup_time": planned_pickup_time, "committed": planned_pickup_time is not None and planned_pickup_time < commit_end and task.status == "Complete", "drop_penalty": total_adddisjunction_cost(batches[task.batch_id], fare_matrix, REVENUE, hs)})
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
    return_to_depots(vehicles, distance, flight_time, nodes, logs)
    if any(b.status == "Pending" for b in batches.values()):
        raise RuntimeError("최종 시뮬레이션에 Pending batch가 남았습니다")
    route_df = pd.DataFrame(logs)
    if not route_df.empty:
        route_df = route_df.sort_values(["vehicle_id", "departure_time", "arrival_time"]).reset_index(drop=True)
        route_df["sequence"] = route_df.groupby("vehicle_id").cumcount()
        """vehicle_route_legs.csv의 분 단위 departure_time/arrival_time은 유지하고, BASE_TIME 기준 실제 시각 열을 추가한다."""  
        route_df["depT_hhmm"] = route_df["departure_time"].map(lambda value: format_hhmm(int(value)))
        route_df["arrT_hhmm"] = route_df["arrival_time"].map(lambda value: format_hhmm(int(value)))
        """depT_hhmm/arrT_hhmm 열에 실제 출발·도착 시각(HH:MM)을 저장한다.""" 

    """내부 route_df는 그대로 두고, 사람이 읽는 vehicle_route_legs.csv 전용 출력 복사본만 만든다. Solver/penalty/occupancy/vehicle summary에는 영향을 주지 않는다."""
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

    status_df = batch_dataframe(batches, maximum_max_wait, fare_matrix)
    vehicle_summary_df = vehicle_summary_dataframe(route_df, vehicles, batches, hs, fare_matrix, task)
    total_operating_cost = int(
        pd.to_numeric(vehicle_summary_df["Total_Cost"], errors="raise").sum()
    ) if not vehicle_summary_df.empty else 0
    total_overdue_cost = int(
        pd.to_numeric(
            status_df.loc[status_df["final_status"].eq("Overdue"), "final_cost(원)"],
            errors="raise",
        ).sum()
    )
    demand_summary_df["total_objectives"] = total_operating_cost + total_overdue_cost

    save_csv(status_df, "rolling_horizon_request_status.csv")
    save_csv(route_output_df, "vehicle_route_legs.csv")
    save_csv(pd.DataFrame(plans), "rolling_horizon_plan_history.csv")
    save_csv(pd.DataFrame(summaries), "rolling_horizon_summary.csv")
    # [추가 수정 4] 물리적 차량 위치 복원을 위해 초기 home_depot 정보가 있는 vehicles를 함께 전달한다.
    save_csv(occupancy_dataframe(route_df, nodes, vehicles), "node_vehicle_occupancy_by_minute.csv")
    # [추가 수정 4 끝] 출력 파일명과 다른 출력 로직은 그대로 유지한다.
    save_csv(vehicle_summary_df, "vehicle_summary.csv")
    save_csv(demand_summary_df,"demand_summary.csv")
    print(f"Complete={sum(b.status == 'Complete' for b in batches.values()):,}, Overdue={sum(b.status == 'Overdue' for b in batches.values()):,}")
    print(f"runtime={time.perf_counter() - started:.2f}s, output={OUTPUT_DIR}")


if __name__ == "__main__":
    main()

