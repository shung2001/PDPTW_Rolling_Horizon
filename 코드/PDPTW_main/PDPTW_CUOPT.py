# -*- coding: utf-8 -*-
"""NVIDIA cuOpt rolling-horizon PDPTW simulation.

Times are integer minutes from BASE_TIME. cuOpt solves the routing/PDP/TW/capacity
subproblem in each rolling horizon. Battery range and charging are preserved as
state across horizons and validated by an outer repair loop because cuOpt Routing
does not expose OR-Tools-style arbitrary cumul/slack equations.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import cudf
    from cuopt import routing
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "NVIDIA cuOpt Python API가 필요합니다. CUDA 버전에 맞는 cuopt 패키지를 설치하세요."
    ) from exc


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_DIR / "자료" / "기초자료"
REQUEST_PATH = INPUT_DIR / "finalDemand_v5" / "finalDemand_v5" / "d5000_s20.csv"
DISTANCE_MATRIX_PATH = INPUT_DIR / "distance_matrix_km.csv"
TIME_MATRIX_PATH = INPUT_DIR / "flight_time_matrix_min.csv"
NODE_REFERENCE_PATH = INPUT_DIR / "vp_reference.csv"
TRANSPORTATION_MATRIX_PATH = PROJECT_DIR / "자료" / "결과" / "차량_교통수단" /"public_transit_time_matrix_min.csv"
OUTPUT_DIR = PROJECT_DIR / "자료" / "결과" / "CuOpt"

BASE_TIME = "05:40"
NUM_VEHICLES = 251
VEHICLE_CAPACITY = 3
DEPOT_ROUTE_NODE_IDS = list(range(1, 11))
ROLLING_HORIZON_MINUTES = 30
REOPTIMIZATION_INTERVAL_MINUTES = 20
TIME_LIMIT_SECONDS = 5

SERVICE_TIME_MINUTES = 3
BOARDING_CHARGE_MINUTES = 2
TAXI_TIME_MINUTES = 1
BASE_DROP_PENALTY = 1_000_000
PENDING_PENALTY_PER_COUNT = 10_000
MAX_WAIT_PENALTY_WEIGHT = 1_000
TRANSPORTATION_JOBY_PENALTY_WEIGHT = 10_000
INITIAL_REMAINING_RANGE_KM = 160.0
MAX_REMAINING_RANGE_KM = 160.0
MIN_REMAINING_RANGE_KM = 15.0
LOW_RANGE_CHARGE_TARGET_KM = 100.0
CHARGING_RATE_KM_PER_MIN = 4.267
LOG_SEARCH = False

# cuOpt-specific controls
CUOPT_PRIZE_OBJECTIVE_WEIGHT = 1.0
CUOPT_COST_OBJECTIVE_WEIGHT = 1.0
CUOPT_MAX_REPAIR_ROUNDS = 40
CUOPT_BLOCKED_ARC_COST = 1.0e9
CUOPT_BLOCKED_ARC_TIME = 1.0e6
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
    data_model: Any
    active: list[RequestTask]
    vehicles: list[VehicleState]
    node_ids: list[str]
    node_to_location: dict[str, int]
    order_to_event: dict[int, tuple[str, RequestTask]]
    hs: int
    he: int
    model_end: int
    return_to_home: bool


@dataclass
class HorizonSolution:
    model: HorizonModel | None
    assignment: Any | None
    route_events: list[dict[str, Any]]
    selected: set[str]
    objective: float | None
    status: int | None
    repair_rounds: int


def solution_status_code(status: Any) -> int | None:
    """Return a stable integer code for both old and new cuOpt status APIs."""
    if status is None:
        return None
    return int(getattr(status, "value", status))


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

# BASE_TIME 기준 분 단위 변환
def base_time_minutes() -> int:
    h, m = map(int, BASE_TIME.split(":")[:2])
    return h * 60 + m


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
    return int(BASE_DROP_PENALTY + PENDING_PENALTY_PER_COUNT * batch.pending_count + MAX_WAIT_PENALTY_WEIGHT * (maximum_max_wait - representative.max_wait_min) + TRANSPORTATION_JOBY_PENALTY_WEIGHT * (representative.transportation_min - joby_min))


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
def _series(values: list[Any] | np.ndarray, dtype: str | None = None):
    """Create a cuDF Series with an optional dtype."""
    return cudf.Series(values, dtype=dtype) if dtype else cudf.Series(values)


def _build_matrix_arrays(
    distance: pd.DataFrame,
    flight_time: pd.DataFrame,
    node_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build float32 cost/time matrices and discourage physically impossible legs."""
    distance_km = distance.loc[node_ids, node_ids].to_numpy(dtype=np.float32, copy=True)
    # Preserve the OR-Tools objective scale: its arc cost was distance[km] * 1000.
    cost = distance_km * np.float32(1000.0)
    transit = flight_time.loc[node_ids, node_ids].to_numpy(dtype=np.float32, copy=True)

    max_leg_km = MAX_REMAINING_RANGE_KM - MIN_REMAINING_RANGE_KM
    impossible = distance_km > (max_leg_km + 1e-9)
    np.fill_diagonal(impossible, False)
    cost[impossible] = np.float32(CUOPT_BLOCKED_ARC_COST)
    transit[impossible] = np.float32(CUOPT_BLOCKED_ARC_TIME)
    return cost, transit


def build_horizon_model(
    active: list[RequestTask],
    batches: dict[str, BatchState],
    vehicles: list[VehicleState],
    distance: pd.DataFrame,
    flight_time: pd.DataFrame,
    hs: int,
    he: int,
    maximum_max_wait: int,
    return_to_home: bool = False,
) -> HorizonModel:
    """Build one cuOpt PDPTW model for the current rolling horizon."""
    if not active:
        raise ValueError("active request가 비어 있습니다")
    if not vehicles:
        raise ValueError("cuOpt horizon에는 차량이 1대 이상 필요합니다")

    node_ids = list(distance.index)
    if set(node_ids) != set(flight_time.index):
        raise ValueError("distance/time matrix node 집합이 다릅니다")
    node_to_location = {node_id: idx for idx, node_id in enumerate(node_ids)}

    n_tasks = len(active)
    n_orders = n_tasks * 2
    vehicle_count = len(vehicles)
    model_end = he + int(flight_time.to_numpy().max()) * 4 + SERVICE_TIME_MINUTES * 4

    data_model = routing.DataModel(len(node_ids), vehicle_count, n_orders)
    cost_matrix, transit_matrix = _build_matrix_arrays(distance, flight_time, node_ids)
    data_model.add_cost_matrix(cudf.DataFrame(cost_matrix))
    data_model.add_transit_time_matrix(cudf.DataFrame(transit_matrix))

    # Vehicle start = current RH vehicle position, return = original home depot.
    starts = np.asarray(
        [node_to_location[v.route_node_id] for v in vehicles], dtype=np.int32
    )
    returns = np.asarray(
        [node_to_location[v.home_depot] for v in vehicles], dtype=np.int32
    )
    data_model.set_vehicle_locations(_series(starts, "int32"), _series(returns, "int32"))
    data_model.set_vehicle_time_windows(
        _series([max(hs, v.available_time) for v in vehicles], "int32"),
        _series([model_end] * vehicle_count, "int32"),
    )
    # Intermediate RH: open end. Final serviceable RH: include return-to-home in solve.
    data_model.set_drop_return_trips(
        _series([not return_to_home] * vehicle_count, "bool")
    )

    pickup_locations = np.asarray(
        [node_to_location[t.origin] for t in active], dtype=np.int32
    )
    delivery_locations = np.asarray(
        [node_to_location[t.destination] for t in active], dtype=np.int32
    )
    data_model.set_order_locations(
        _series(np.concatenate([pickup_locations, delivery_locations]), "int32")
    )

    pickup_orders = np.arange(n_tasks, dtype=np.int32)
    delivery_orders = np.arange(n_tasks, 2 * n_tasks, dtype=np.int32)
    data_model.set_pickup_delivery_pairs(
        _series(pickup_orders, "int32"),
        _series(delivery_orders, "int32"),
    )

    # Pickup adds passengers, delivery removes them.
    demand = np.asarray(
        [t.passengers for t in active] + [-t.passengers for t in active],
        dtype=np.int32,
    )
    capacity = np.asarray([VEHICLE_CAPACITY] * vehicle_count, dtype=np.int32)
    data_model.add_capacity_dimension(
        "passengers",
        _series(demand, "int32"),
        _series(capacity, "int32"),
    )

    earliest_pickup = [max(hs, t.departure_time) for t in active]
    latest_pickup = [he for _ in active]
    earliest_delivery = [max(hs, t.arrival_time) for t in active]
    latest_delivery = [min(model_end, t.window_end) for t in active]
    earliest = np.asarray(earliest_pickup + earliest_delivery, dtype=np.int32)
    latest = np.asarray(latest_pickup + latest_delivery, dtype=np.int32)
    if np.any(earliest > latest):
        bad = np.where(earliest > latest)[0].tolist()
        raise ValueError(f"cuOpt order time window가 역전되었습니다: order_indices={bad}")
    data_model.set_order_time_windows(
        _series(earliest, "int32"),
        _series(latest, "int32"),
    )
    data_model.set_order_service_times(
        _series([SERVICE_TIME_MINUTES] * n_orders, "int32")
    )

    # Pair must be accepted/dropped together. Give half of the batch drop penalty
    # to each event so a served P-D pair collects approximately one full batch prize.
    pair_prizes = [
        float(drop_penalty(batches[t.batch_id], maximum_max_wait)) / 2.0
        for t in active
    ]
    data_model.set_order_prizes(
        _series(pair_prizes + pair_prizes, "float32")
    )
    data_model.set_objective_function(
        _series([routing.Objective.PRIZE, routing.Objective.COST]),
        _series(
            [CUOPT_PRIZE_OBJECTIVE_WEIGHT, CUOPT_COST_OBJECTIVE_WEIGHT],
            dtype="float32",
        ),
    )

    order_to_event: dict[int, tuple[str, RequestTask]] = {}
    for i, task in enumerate(active):
        order_to_event[i] = ("Pickup", task)
        order_to_event[i + n_tasks] = ("Delivery", task)

    return HorizonModel(
        data_model=data_model,
        active=active,
        vehicles=vehicles,
        node_ids=node_ids,
        node_to_location=node_to_location,
        order_to_event=order_to_event,
        hs=hs,
        he=he,
        model_end=model_end,
        return_to_home=return_to_home,
    )


def solve_horizon(model: HorizonModel):
    """Run cuOpt once for one already-built horizon model."""
    settings = routing.SolverSettings()
    settings.set_time_limit(float(TIME_LIMIT_SECONDS))
    settings.set_verbose_mode(bool(LOG_SEARCH))
    assignment = routing.Solve(model.data_model, settings)
    return assignment


def _assignment_route_to_pandas(assignment: Any) -> pd.DataFrame:
    route = assignment.get_route()
    if route is None:
        return pd.DataFrame()
    if hasattr(route, "to_pandas"):
        return route.to_pandas()
    return pd.DataFrame(route)


def extract_routes(
    model: HorizonModel,
    assignment: Any,
    vehicles: list[VehicleState],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Convert cuOpt route rows into the legacy Pickup/Delivery event format."""
    if assignment is None:
        return [], set()
    status = solution_status_code(assignment.get_status())
    if status != solution_status_code(routing.SolutionStatus.SUCCESS):
        return [], set()

    route_df = _assignment_route_to_pandas(assignment)
    if route_df.empty:
        return [], set()

    required = {"route", "arrival_stamp", "truck_id", "type"}
    missing = required - set(route_df.columns)
    if missing:
        raise KeyError(
            f"cuOpt get_route() 결과에 필요한 열이 없습니다: missing={sorted(missing)}, "
            f"actual={list(route_df.columns)}"
        )

    rows: list[dict[str, Any]] = []
    selected: set[str] = set()
    sequence_by_vehicle: dict[int, int] = {}

    for _, row in route_df.iterrows():
        type_text = str(row["type"]).strip().lower()
        if type_text in {"start", "end", "depot", "break"}:
            continue
        if "pickup" not in type_text and "delivery" not in type_text:
            continue

        try:
            order_idx = int(row["route"])
        except (TypeError, ValueError):
            continue
        if order_idx not in model.order_to_event:
            # Start/end rows can use a location id in the route column.
            continue

        event, task = model.order_to_event[order_idx]
        if ("pickup" in type_text and event != "Pickup") or (
            "delivery" in type_text and event != "Delivery"
        ):
            raise RuntimeError(
                "cuOpt route 열이 order index로 해석되지 않습니다. "
                f"route={order_idx}, type={row['type']}, expected={event}"
            )
        vehicle_id = int(row["truck_id"])
        seq = sequence_by_vehicle.get(vehicle_id, 0)
        sequence_by_vehicle[vehicle_id] = seq + 1
        event_row = {
            "vehicle_id": vehicle_id,
            "sequence": seq,
            "event": event,
            "task": task,
            "time": int(round(float(row["arrival_stamp"]))),
            "cuopt_order_index": order_idx,
        }
        rows.append(event_row)
        if event == "Pickup":
            selected.add(task.task_key)

    rows.sort(key=lambda r: (int(r["vehicle_id"]), int(r["sequence"])))
    return rows, selected


def _alternative_winners(
    route_events: list[dict[str, Any]],
) -> dict[str, str]:
    """For batches with >1 selected alternative, choose one deterministic winner."""
    by_batch: dict[str, dict[str, dict[str, float]]] = {}
    for row in route_events:
        task: RequestTask = row["task"]
        info = by_batch.setdefault(task.batch_id, {}).setdefault(
            task.task_key,
            {
                "pickup": float("inf"),
                "delivery": float("inf"),
                "distance": float(task.distance_km),
            },
        )
        if row["event"] == "Pickup":
            info["pickup"] = min(info["pickup"], float(row["time"]))
        else:
            info["delivery"] = min(info["delivery"], float(row["time"]))

    winners: dict[str, str] = {}
    for batch_id, alternatives in by_batch.items():
        if len(alternatives) <= 1:
            continue
        winner = min(
            alternatives,
            key=lambda key: (
                alternatives[key]["delivery"],
                alternatives[key]["pickup"],
                alternatives[key]["distance"],
                key,
            ),
        )
        winners[batch_id] = winner
    return winners


def _reschedule_routes_with_battery(
    route_events: list[dict[str, Any]],
    vehicles: list[VehicleState],
    distance: pd.DataFrame,
    flight_time: pd.DataFrame,
    hs: int,
    he: int,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Respect cuOpt order sequence but recompute exact times with charging/range state.

    Returns a battery-aware event schedule and task keys that make the sequence
    infeasible. Those tasks are removed and the horizon is re-solved by cuOpt.
    """
    state_by_id = {v.vehicle_id: v for v in vehicles}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in route_events:
        grouped.setdefault(int(row["vehicle_id"]), []).append(row)

    scheduled: list[dict[str, Any]] = []
    invalid: set[str] = set()

    for vehicle_id, events in grouped.items():
        events.sort(key=lambda r: int(r["sequence"]))
        state = state_by_id[vehicle_id]
        current_node = state.route_node_id
        current_time = state.available_time
        remaining = state.remaining_range_km

        # Time before this RH can be used for charging, but no new departure
        # is allowed before hs.
        if current_time < hs:
            idle = hs - current_time
            remaining = min(
                MAX_REMAINING_RANGE_KM,
                remaining + idle * CHARGING_RATE_KM_PER_MIN,
            )
            current_time = hs

        for row in events:
            task: RequestTask = row["task"]
            event = str(row["event"])
            target = task.origin if event == "Pickup" else task.destination
            dist = float(distance.loc[current_node, target])
            travel = int(flight_time.loc[current_node, target])

            if dist > MAX_REMAINING_RANGE_KM - MIN_REMAINING_RANGE_KM + 1e-9:
                invalid.add(task.task_key)
                break

            if event == "Pickup":
                earliest = max(hs, task.departure_time)
                latest = he
            else:
                earliest = max(hs, task.arrival_time)
                latest = task.window_end

            # Wait at the current node if arriving too early; waiting charges battery.
            natural_arrival = current_time + travel
            wait = max(0, earliest - natural_arrival)
            if wait:
                remaining = min(
                    MAX_REMAINING_RANGE_KM,
                    remaining + wait * CHARGING_RATE_KM_PER_MIN,
                )
                current_time += wait

            try:
                extra_charge, departure_range = charge_for_departure(remaining, dist)
            except RuntimeError:
                invalid.add(task.task_key)
                break

            current_time += extra_charge
            departure = current_time
            arrival = departure + travel
            if arrival > latest:
                invalid.add(task.task_key)
                break

            after_flight = departure_range - dist
            service_charge = min(
                MAX_REMAINING_RANGE_KM,
                after_flight + BOARDING_CHARGE_MINUTES * CHARGING_RATE_KM_PER_MIN,
            )
            new_row = dict(row)
            new_row["time"] = int(arrival)
            scheduled.append(new_row)

            current_node = target
            current_time = int(arrival) + SERVICE_TIME_MINUTES
            remaining = service_charge

    scheduled.sort(key=lambda r: (int(r["vehicle_id"]), int(r["sequence"])))
    return scheduled, invalid


def solve_horizon_with_repairs(
    active: list[RequestTask],
    batches: dict[str, BatchState],
    vehicles: list[VehicleState],
    distance: pd.DataFrame,
    flight_time: pd.DataFrame,
    hs: int,
    he: int,
    maximum_max_wait: int,
    return_to_home: bool,
) -> HorizonSolution:
    """Solve a cuOpt horizon and repair unsupported custom constraints externally.

    Repair 1: at most one alternative request per batch.
    Repair 2: battery/charging-aware timing using persistent VehicleState.
    """
    forbidden: set[str] = set()
    locked_choice: dict[str, str] = {}
    last_assignment: Any | None = None
    last_model: HorizonModel | None = None

    for repair_round in range(1, CUOPT_MAX_REPAIR_ROUNDS + 1):
        candidates = [
            task
            for task in active
            if task.task_key not in forbidden
            and (
                task.batch_id not in locked_choice
                or locked_choice[task.batch_id] == task.task_key
            )
        ]
        if not candidates:
            return HorizonSolution(
                model=None,
                assignment=None,
                route_events=[],
                selected=set(),
                objective=None,
                status=None,
                repair_rounds=repair_round,
            )

        model = build_horizon_model(
            candidates,
            batches,
            vehicles,
            distance,
            flight_time,
            hs,
            he,
            maximum_max_wait,
            return_to_home,
        )
        assignment = solve_horizon(model)
        last_assignment, last_model = assignment, model
        status = (
            solution_status_code(assignment.get_status())
            if assignment is not None
            else None
        )
        if assignment is None or status != solution_status_code(
            routing.SolutionStatus.SUCCESS
        ):
            return HorizonSolution(
                model=model,
                assignment=assignment,
                route_events=[],
                selected=set(),
                objective=None,
                status=status,
                repair_rounds=repair_round,
            )

        raw_routes, raw_selected = extract_routes(model, assignment, vehicles)

        # cuOpt has no AddDisjunction(..., max_cardinality=1) equivalent for an
        # arbitrary batch of alternatives. If multiple alternatives of one batch
        # were selected, lock the best-fitting one and solve again.
        winners = _alternative_winners(raw_routes)
        changed = False
        for batch_id, winner in winners.items():
            if locked_choice.get(batch_id) != winner:
                locked_choice[batch_id] = winner
                changed = True
        if changed:
            continue

        scheduled_routes, invalid = _reschedule_routes_with_battery(
            raw_routes,
            vehicles,
            distance,
            flight_time,
            hs,
            he,
        )
        if invalid:
            for task_key in invalid:
                forbidden.add(task_key)
                task = next((t for t in candidates if t.task_key == task_key), None)
                if task is not None and locked_choice.get(task.batch_id) == task_key:
                    # Allow another alternative from the same batch next round.
                    locked_choice.pop(task.batch_id, None)
            continue

        selected = {
            row["task"].task_key
            for row in scheduled_routes
            if row["event"] == "Pickup"
        }
        objective = float(assignment.get_total_objective())
        return HorizonSolution(
            model=model,
            assignment=assignment,
            route_events=scheduled_routes,
            selected=selected,
            objective=objective,
            status=status,
            repair_rounds=repair_round,
        )

    status = (
        solution_status_code(last_assignment.get_status())
        if last_assignment is not None
        else None
    )
    return HorizonSolution(
        model=last_model,
        assignment=last_assignment,
        route_events=[],
        selected=set(),
        objective=None,
        status=status,
        repair_rounds=CUOPT_MAX_REPAIR_ROUNDS,
    )


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
            remaining = min(MAX_REMAINING_RANGE_KM, remaining + parking * CHARGING_RATE_KM_PER_MIN)
            extra_charge, departure_range = charge_for_departure(remaining, dist)
            departure = current_time + parking + extra_charge
            arrival = departure + travel
            after_flight = departure_range - dist
            before_count = sum(onboard.values())
            related = task.batch_id
            log_rows.append({"vehicle_id": vehicle_id, "sequence": len(log_rows), "event": "Movement", "from_node": current_node, "to_node": target, "physical_from_node": nodes[current_node].physical_node_id, "physical_to_node": nodes[target].physical_node_id, "departure_time": departure, "arrival_time": arrival, "travel_time": travel, "travel_distance_km": round(dist, 3), "charging_time": parking + extra_charge, "charged_range_km": round(departure_range - state.remaining_range_km if current_node == state.route_node_id else parking * CHARGING_RATE_KM_PER_MIN + extra_charge * CHARGING_RATE_KM_PER_MIN, 3), "remaining_range_before": round(remaining, 3), "remaining_range_after": round(after_flight, 3), "onboard_passengers": before_count, "related_batch_id": related, "movement_type": "passenger_flight" if before_count else "empty_repositioning", "from_event": "Position", "to_event": event})
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
        logs.append({"vehicle_id": state.vehicle_id, "sequence": len(logs), "event": "Movement", "from_node": state.route_node_id, "to_node": state.home_depot, "physical_from_node": nodes[state.route_node_id].physical_node_id, "physical_to_node": nodes[state.home_depot].physical_node_id, "departure_time": departure, "arrival_time": arrival, "travel_time": arrival - departure, "travel_distance_km": round(dist, 3), "charging_time": charge_min, "charged_range_km": round(departure_range - state.remaining_range_km, 3), "remaining_range_before": round(state.remaining_range_km, 3), "remaining_range_after": round(departure_range - dist, 3), "onboard_passengers": 0, "related_batch_id": "", "movement_type": "return_to_depot", "from_event": "Position", "to_event": "Depot"})
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
        rows.append({"batch_id": batch.batch_id, "selected_request_id": batch.selected_request_id, "origin": selected.origin, "destination": selected.destination, "max_wait": selected.max_wait_min, "Transportation_min": selected.transportation_min, "Joby_min": selected.joby_min, "Transportation_Joby_time_saving": selected.transportation_min - selected.joby_min, "pending_count": batch.pending_count, "final_drop_penalty": drop_penalty(batch, maximum_max_wait), "assigned_vehicle": batch.assigned_vehicle_id, "pickup_time": batch.pickup_time, "pickup_time_hhmm": format_hhmm(batch.pickup_time), "delivery_time": batch.delivery_time, "delivery_time_hhmm": format_hhmm(batch.delivery_time), "final_status": batch.status})
    return pd.DataFrame(rows)


def occupancy_dataframe(route_df: pd.DataFrame, nodes: dict[str, NodeInfo]) -> pd.DataFrame:
    """Preserve the legacy minute-level vertiport occupancy output."""
    rows: list[dict[str, Any]] = []
    if route_df.empty:
        return pd.DataFrame(columns=["time_min", "time_hhmm", "route_node_id", "physical_node_id", "physical_address", "vehicle_count", "vehicle_ids"])
    stops = route_df[route_df["event"].isin(["Pickup", "Delivery"])]
    for _, row in stops.iterrows():
        for minute in range(int(row["arrival_time"]), int(row["departure_time"])):
            rows.append({"time_min": minute, "route_node_id": str(row["to_node"]), "vehicle_id": int(row["vehicle_id"])})
    if not rows:
        return pd.DataFrame(columns=["time_min", "time_hhmm", "route_node_id", "physical_node_id", "physical_address", "vehicle_count", "vehicle_ids"])
    result = pd.DataFrame(rows).groupby(["time_min", "route_node_id"], as_index=False).agg(vehicle_count=("vehicle_id", "nunique"), vehicle_ids=("vehicle_id", lambda values: ",".join(map(str, sorted(set(values))))))
    result["time_hhmm"] = result["time_min"].map(format_hhmm)
    result["physical_node_id"] = result["route_node_id"].map(lambda value: nodes[value].physical_node_id)
    result["physical_address"] = result["route_node_id"].map(lambda value: nodes[value].address)
    return result[["time_min", "time_hhmm", "route_node_id", "physical_node_id", "physical_address", "vehicle_count", "vehicle_ids"]]


def vehicle_summary_dataframe(route_df: pd.DataFrame, vehicles: list[VehicleState]) -> pd.DataFrame:
    rows = []
    for state in vehicles:
        vr = route_df[route_df["vehicle_id"] == state.vehicle_id] if not route_df.empty else pd.DataFrame()
        movement = vr[vr["event"] == "Movement"] if not vr.empty else pd.DataFrame()
        passenger = movement[movement["movement_type"] == "passenger_flight"] if not movement.empty else pd.DataFrame()
        empty = movement[movement["movement_type"].isin(["empty_repositioning", "return_to_depot"])] if not movement.empty else pd.DataFrame()
        rows.append({"vehicle_id": state.vehicle_id, "home_depot": state.home_depot, "final_node": state.route_node_id, "final_available_time": state.available_time, "final_remaining_range_km": round(state.remaining_range_km, 3), "total_flight_time_min": int(movement["travel_time"].sum()) if not movement.empty else 0, "total_flight_distance_km": round(float(movement["travel_distance_km"].sum()), 3) if not movement.empty else 0.0, "passenger_flight_distance_km": round(float(passenger["travel_distance_km"].sum()), 3) if not passenger.empty else 0.0, "empty_flight_distance_km": round(float(empty["travel_distance_km"].sum()), 3) if not empty.empty else 0.0, "charging_time_min": int(vr["charging_time"].sum()) if not vr.empty else 0, "pickup_count": int((vr["event"] == "Pickup").sum()) if not vr.empty else 0, "delivery_count": int((vr["event"] == "Delivery").sum()) if not vr.empty else 0})
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
    simulation_end = max(t.window_end for t in tasks) + int(flight_time.to_numpy().max()) + SERVICE_TIME_MINUTES
    horizon_starts = range(0, simulation_end + 1, REOPTIMIZATION_INTERVAL_MINUTES)
    total_horizons = len(horizon_starts)
    seen_batch_ids: set[str] = set()

    for rh_index, hs in enumerate(horizon_starts, start=1):
        update_batch_states(batches, hs)
        he = hs + ROLLING_HORIZON_MINUTES
        commit_end = hs + REOPTIMIZATION_INTERVAL_MINUTES
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
            print(f"최적화 구간    : {format_hhmm(hs)} ~ {format_hhmm(he)}  ({ROLLING_HORIZON_MINUTES}분)", flush=True)
            print(f"확정/Commit 구간: {format_hhmm(hs)} ~ {format_hhmm(commit_end)}  ({REOPTIMIZATION_INTERVAL_MINUTES}분)", flush=True)
            print(f"재계획 중첩구간 : {format_hhmm(overlap_start)} ~ {format_hhmm(overlap_end)}  ({ROLLING_HORIZON_MINUTES - REOPTIMIZATION_INTERVAL_MINUTES}분)", flush=True)
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
            return_to_home = commit_end > latest_window_end
            horizon_solution = solve_horizon_with_repairs(
                active,
                batches,
                vehicles,
                distance,
                flight_time,
                hs,
                he,
                maximum_max_wait,
                return_to_home,
            )
            """변경 끝"""
            runtime = time.perf_counter() - tick
            routes = horizon_solution.route_events
            selected = horizon_solution.selected
            objective = (
                None
                if horizon_solution.objective is None
                else int(round(horizon_solution.objective))
            )
            committed = commit_routes(routes, vehicles, distance, flight_time, nodes, commit_end, logs, batches)
            for task in active:
                plans.append({"horizon_start": hs, "horizon_end": he, "request_id": task.request_id, "batch_id": task.batch_id, "planned_visit": task.task_key in selected, "drop_penalty": drop_penalty(batches[task.batch_id], maximum_max_wait)})
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
    return_to_depots(vehicles, distance, flight_time, nodes, logs)
    if any(b.status == "Pending" for b in batches.values()):
        raise RuntimeError("최종 시뮬레이션에 Pending batch가 남았습니다")
    route_df = pd.DataFrame(logs)
    if not route_df.empty:
        route_df = route_df.sort_values(["vehicle_id", "departure_time", "arrival_time"]).reset_index(drop=True)
        route_df["sequence"] = route_df.groupby("vehicle_id").cumcount()
    save_csv(batch_dataframe(batches, maximum_max_wait), "rolling_horizon_request_status.csv")
    save_csv(route_df, "vehicle_route_legs.csv")
    save_csv(pd.DataFrame(plans), "rolling_horizon_plan_history.csv")
    save_csv(pd.DataFrame(summaries), "rolling_horizon_summary.csv")
    save_csv(occupancy_dataframe(route_df, nodes), "node_vehicle_occupancy_by_minute.csv")
    save_csv(vehicle_summary_dataframe(route_df, vehicles), "vehicle_summary.csv")
    print(f"Complete={sum(b.status == 'Complete' for b in batches.values()):,}, Overdue={sum(b.status == 'Overdue' for b in batches.values()):,}")
    print(f"runtime={time.perf_counter() - started:.2f}s, output={OUTPUT_DIR}")


if __name__ == "__main__":
    main()



