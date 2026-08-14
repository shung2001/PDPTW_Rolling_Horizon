"""Kakao 자동차 길찾기를 OD(출발지-도착지) 한 쌍씩 호출해 행렬을 만든다.

다중 목적지 API를 사용하지 않는다. 각 요청은 ``GET /v1/directions``이며,
성공할 때마다 CSV를 저장하므로 실행이 중단되어도 ``--resume``으로 이어갈 수 있다.

실행 예:
    $env:KAKAO_REST_API_KEY="발급받은_REST_API_KEY"
    python "코드/Time_matrix_From_Kakao_Individual.py" --resume
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://apis-navi.kakaomobility.com/v1/directions"
KAKAO_REST_API_KEY = "f1161b2a855cf332983a74de75da4167"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_DIR / "자료" / "기초자료" / "vp_reference.csv"
DEFAULT_OUTPUT_SEC = PROJECT_DIR / "자료" / "결과" / "driving_time_matrix_individual_sec.csv"
DEFAULT_OUTPUT_MIN = PROJECT_DIR / "자료" / "결과" / "driving_time_matrix_individual_min.csv"
DEFAULT_OUTPUT_KM = PROJECT_DIR / "자료" / "결과" / "driving_distance_matrix_individual_km.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kakao 자동차 길찾기를 OD별로 호출하여 시간/거리 행렬 생성"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-sec", type=Path, default=DEFAULT_OUTPUT_SEC)
    parser.add_argument("--output-min", type=Path, default=DEFAULT_OUTPUT_MIN)
    parser.add_argument("--output-km", type=Path, default=DEFAULT_OUTPUT_KM)
    parser.add_argument("--priority", choices=("RECOMMEND", "TIME", "DISTANCE"), default="TIME")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--request-interval", type=float, default=0.15)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="기존 초/거리 행렬의 값은 건너뛰고 빈 OD만 다시 계산",
    )
    return parser.parse_args()


def read_points(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    text = None
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            pass
    if text is None:
        raise ValueError(f"CSV 인코딩을 해석하지 못했습니다: {path}")

    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    if not rows:
        raise ValueError(f"입력 CSV가 비어 있습니다: {path}")
    missing = {"lat", "lon"} - set(rows[0])
    if missing:
        raise ValueError(f"입력 CSV 필수 열 누락: {', '.join(sorted(missing))}")

    points: list[dict[str, Any]] = []
    labels: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{line_number}행의 lat/lon 값이 올바르지 않습니다") from exc
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"{line_number}행 좌표 범위 오류: lat={lat}, lon={lon}")

        label = str(row.get("vpname") or row.get("vp_id") or len(points)).strip()
        if not label or label in labels:
            raise ValueError(f"{line_number}행의 지점명이 비었거나 중복되었습니다: {label!r}")
        labels.add(label)
        points.append({"label": label, "lat": lat, "lon": lon})
    return points


def request_route(
    origin: dict[str, Any],
    destination: dict[str, Any],
    api_key: str,
    priority: str,
    retries: int,
) -> tuple[float, float]:
    query = urlencode(
        {
            "origin": f"{origin['lon']},{origin['lat']}",
            "destination": f"{destination['lon']},{destination['lat']}",
            "priority": priority,
            "summary": "true",
        }
    )
    request = Request(
        f"{API_URL}?{query}",
        headers={"Authorization": f"KakaoAK {api_key}", "Content-Type": "application/json"},
        method="GET",
    )

    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            routes = result.get("routes") or []
            if not routes:
                raise RuntimeError(f"경로가 없는 응답: {result}")
            route = routes[0]
            if route.get("result_code") != 0 or not route.get("summary"):
                raise RuntimeError(
                    f"길찾기 실패 [{route.get('result_code')}]: {route.get('result_msg')}"
                )
            summary = route["summary"]
            return float(summary["duration"]), float(summary["distance"]) / 1000.0
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            error: Exception = RuntimeError(f"Kakao API HTTP {exc.code}: {detail}")
            if exc.code < 500 and exc.code != 429:
                raise error from exc
        except (URLError, TimeoutError, json.JSONDecodeError, KeyError, RuntimeError) as exc:
            error = exc

        if attempt == retries:
            raise RuntimeError(f"{retries + 1}회 요청 실패: {error}") from error
        wait_seconds = 2**attempt
        print(f"    요청 실패, {wait_seconds}초 후 재시도 ({attempt + 1}/{retries})", file=sys.stderr)
        time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def empty_matrix(size: int) -> list[list[float]]:
    matrix = [[math.nan] * size for _ in range(size)]
    for index in range(size):
        matrix[index][index] = 0.0
    return matrix


def load_matrix(path: Path, labels: list[str]) -> list[list[float]]:
    matrix = empty_matrix(len(labels))
    if not path.exists():
        return matrix
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    if not rows or rows[0][1:] != labels or [row[0] for row in rows[1:]] != labels:
        raise ValueError(f"이어받을 행렬의 지점 순서가 현재 입력과 다릅니다: {path}")
    if len(rows) != len(labels) + 1:
        raise ValueError(f"이어받을 행렬 크기가 올바르지 않습니다: {path}")
    for i, row in enumerate(rows[1:]):
        if len(row) != len(labels) + 1:
            raise ValueError(f"이어받을 행렬 {i + 2}행의 열 개수가 올바르지 않습니다: {path}")
        for j, value in enumerate(row[1:]):
            if value.strip():
                matrix[i][j] = float(value)
    return matrix


def write_matrix(path: Path, labels: list[str], matrix: list[list[float]], unit: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["origin\\destination", *labels])
        for label, row in zip(labels, matrix):
            values: list[str | int | float] = []
            for value in row:
                if math.isnan(value):
                    values.append("")
                elif unit == "seconds":
                    values.append(int(round(value)))
                elif unit == "minutes":
                    values.append(round(value / 60.0, 2))
                else:
                    values.append(round(value, 3))
            writer.writerow([label, *values])
    temporary.replace(path)


def save_all(
    args: argparse.Namespace,
    labels: list[str],
    seconds: list[list[float]],
    distances: list[list[float]],
) -> None:
    write_matrix(args.output_sec.resolve(), labels, seconds, "seconds")
    write_matrix(args.output_min.resolve(), labels, seconds, "minutes")
    write_matrix(args.output_km.resolve(), labels, distances, "kilometres")


def main() -> None:
    args = parse_args()
    if args.retries < 0 or args.request_interval < 0:
        raise ValueError("retries와 request-interval은 0 이상이어야 합니다")
    api_key = os.environ.get("KAKAO_REST_API_KEY", KAKAO_REST_API_KEY).strip()
    if not api_key:
        raise RuntimeError("환경 변수 KAKAO_REST_API_KEY를 설정해 주세요")

    points = read_points(args.input.resolve())
    labels = [point["label"] for point in points]
    seconds = load_matrix(args.output_sec.resolve(), labels) if args.resume else empty_matrix(len(points))
    distances = load_matrix(args.output_km.resolve(), labels) if args.resume else empty_matrix(len(points))

    total = len(points) * (len(points) - 1)
    completed = sum(
        1
        for i in range(len(points))
        for j in range(len(points))
        if i != j and not math.isnan(seconds[i][j]) and not math.isnan(distances[i][j])
    )
    failures: list[tuple[str, str, str]] = []
    print(f"총 {total}개 OD 중 {completed}개 완료, {total - completed}개 계산 시작")

    for i, origin in enumerate(points):
        for j, destination in enumerate(points):
            if i == j:
                continue
            if not math.isnan(seconds[i][j]) and not math.isnan(distances[i][j]):
                continue
            print(f"[{completed + 1}/{total}] {origin['label']} -> {destination['label']}")
            try:
                duration_sec, distance_km = request_route(
                    origin, destination, api_key, args.priority, args.retries
                )
                seconds[i][j] = duration_sec
                distances[i][j] = distance_km
                completed += 1
                print(f"    {duration_sec / 60:.2f}분, {distance_km:.3f}km")
                save_all(args, labels, seconds, distances)
            except RuntimeError as exc:
                failures.append((origin["label"], destination["label"], str(exc)))
                print(f"    최종 실패: {exc}", file=sys.stderr)
            if args.request_interval:
                time.sleep(args.request_interval)

    save_all(args, labels, seconds, distances)
    print(f"\n초 단위 시간 행렬: {args.output_sec.resolve()}")
    print(f"분 단위 시간 행렬: {args.output_min.resolve()}")
    print(f"km 단위 거리 행렬: {args.output_km.resolve()}")
    if failures:
        print(f"실패 OD: {len(failures)}개 (재실행 시 --resume으로 재시도 가능)", file=sys.stderr)
        for origin, destination, error in failures:
            print(f"  {origin} -> {destination}: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
