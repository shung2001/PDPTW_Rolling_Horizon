"""TMAP 대중교통 API로 OD 시간/요금 행렬을 생성한다.

입력 CSV에는 ``lat``, ``lon`` 열이 필요하며, 행/열 이름은 ``vpname`` 또는
``vp_id``를 사용한다. 각 요청이 성공할 때마다 결과를 저장하므로 중단 후
``--resume`` 옵션으로 이어서 실행할 수 있다.

실행 예시 (PowerShell):
    TMAP_APP_KEY 환경 변수에 발급받은 AppKey를 지정한 뒤 실행한다.
    python "코드/Time_Cost_matrix_From_Tmap.py" --resume
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
from urllib.request import Request, urlopen


API_URL = "https://apis.openapi.sk.com/transit/routes"
CONFIG = {
    "api_key": "USukMvWK035rjx8kD2vGs7r6y1woFY7k3wx0JPnY",
}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_DIR / "자료" / "기초자료" / "vp_reference.csv"
DEFAULT_OUTPUT_MIN = PROJECT_DIR / "자료" / "결과" / "public_transit_time_matrix_tmap_min.csv"
DEFAULT_OUTPUT_FARE = PROJECT_DIR / "자료" / "결과" / "public_transit_fare_matrix_tmap_krw.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TMAP 대중교통 길찾기를 OD별로 호출하여 시간/요금 행렬 생성"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-min", type=Path, default=DEFAULT_OUTPUT_MIN)
    parser.add_argument("--output-fare", type=Path, default=DEFAULT_OUTPUT_FARE)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("TMAP_APP_KEY", CONFIG.get("api_key", "")),
        help="TMAP AppKey (기본값: TMAP_APP_KEY 환경 변수)",
    )
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--request-interval", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true",
                        help="기존 행렬의 빈 OD만 다시 계산")
    return parser.parse_args()


class TmapAuthenticationError(RuntimeError):
    """TMAP AppKey가 없거나 유효하지 않을 때 발생한다."""


def read_points(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    text = None
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
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
    retries: int,
) -> tuple[float, float]:
    body = json.dumps({
        "startX": str(origin["lon"]),
        "startY": str(origin["lat"]),
        "endX": str(destination["lon"]),
        "endY": str(destination["lat"]),
        "count": 1,
        "lang": 0,
        "format": "json",
    }).encode("utf-8")
    request = Request(
        API_URL,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "appKey": api_key,
        },
        method="POST",
    )

    error: Exception = RuntimeError("알 수 없는 오류")
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            itineraries = data.get("metaData", {}).get("plan", {}).get("itineraries") or []
            if not itineraries:
                raise RuntimeError(f"검색된 대중교통 경로가 없습니다: {data}")
            itinerary = itineraries[0]
            duration_min = float(itinerary["totalTime"]) / 60.0
            fare_krw = float(itinerary["fare"]["regular"]["totalFare"])
            return duration_min, fare_krw
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            error = RuntimeError(f"TMAP API HTTP {exc.code}: {detail}")
            if exc.code in (401, 403):
                raise TmapAuthenticationError(
                    f"TMAP API 인증 실패(HTTP {exc.code}). "
                    "유효한 AppKey를 --api-key 또는 TMAP_APP_KEY로 지정하세요.\n"
                    f"서버 응답: {detail}"
                ) from exc
            if exc.code < 500 and exc.code != 429:
                raise error from exc
        except (URLError, TimeoutError, json.JSONDecodeError, KeyError,
                TypeError, ValueError, RuntimeError) as exc:
            error = exc

        if attempt == retries:
            raise RuntimeError(f"{retries + 1}회 요청 실패: {error}") from error
        wait_seconds = 2 ** attempt
        print(f"    요청 실패, {wait_seconds}초 후 재시도 ({attempt + 1}/{retries})",
              file=sys.stderr)
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
    if (not rows or rows[0][1:] != labels
            or [row[0] for row in rows[1:]] != labels
            or len(rows) != len(labels) + 1):
        raise ValueError(f"이어받을 행렬의 지점 순서/크기가 입력과 다릅니다: {path}")
    for i, row in enumerate(rows[1:]):
        if len(row) != len(labels) + 1:
            raise ValueError(f"행렬 {i + 2}행의 열 개수가 올바르지 않습니다: {path}")
        for j, value in enumerate(row[1:]):
            if value.strip():
                matrix[i][j] = float(value)
    return matrix


def write_matrix(path: Path, labels: list[str], matrix: list[list[float]], digits: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["origin\\destination", *labels])
        for label, row in zip(labels, matrix):
            values = [
                "" if math.isnan(value)
                else int(round(value)) if digits == 0
                else round(value, digits)
                for value in row
            ]
            writer.writerow([label, *values])
    temporary.replace(path)


def save_all(args: argparse.Namespace, labels: list[str], minutes: list[list[float]],
             fares: list[list[float]]) -> None:
    write_matrix(args.output_min.resolve(), labels, minutes, 2)
    write_matrix(args.output_fare.resolve(), labels, fares, 0)


def main() -> None:
    args = parse_args()
    if args.retries < 0 or args.request_interval < 0:
        raise ValueError("retries와 request-interval은 0 이상이어야 합니다")
    api_key = str(args.api_key).strip()
    if not api_key:
        raise SystemExit(
            "TMAP AppKey가 없습니다. --api-key 옵션 또는 TMAP_APP_KEY 환경 변수로 지정하세요."
        )
    points = read_points(args.input.resolve())
    labels = [point["label"] for point in points]
    minutes = load_matrix(args.output_min.resolve(), labels) if args.resume else empty_matrix(len(points))
    fares = load_matrix(args.output_fare.resolve(), labels) if args.resume else empty_matrix(len(points))

    total = len(points) * (len(points) - 1)
    completed = sum(
        1 for i in range(len(points)) for j in range(len(points))
        if i != j and not math.isnan(minutes[i][j]) and not math.isnan(fares[i][j])
    )
    failures: list[tuple[str, str, str]] = []
    print(f"총 {total}개 OD 중 {completed}개 완료, {total - completed}개 계산 시작")

    for i, origin in enumerate(points):
        for j, destination in enumerate(points):
            if (i == j or (not math.isnan(minutes[i][j])
                           and not math.isnan(fares[i][j]))):
                continue
            print(f"[{completed + 1}/{total}] {origin['label']} -> {destination['label']}")
            try:
                duration_min, fare_krw = request_route(
                    origin, destination, api_key, args.retries
                )
                minutes[i][j] = duration_min
                fares[i][j] = fare_krw
                completed += 1
                print(f"    {duration_min:.2f}분, {fare_krw:.0f}원")
                save_all(args, labels, minutes, fares)
            except TmapAuthenticationError as exc:
                save_all(args, labels, minutes, fares)
                raise SystemExit(str(exc)) from exc
            except RuntimeError as exc:
                failures.append((origin["label"], destination["label"], str(exc)))
                print(f"    최종 실패: {exc}", file=sys.stderr)
            if args.request_interval:
                time.sleep(args.request_interval)

    save_all(args, labels, minutes, fares)
    print(f"\n분 단위 시간 행렬: {args.output_min.resolve()}")
    print(f"원 단위 요금 행렬: {args.output_fare.resolve()}")
    if failures:
        print(f"실패 OD: {len(failures)}개 (--resume으로 재시도 가능)", file=sys.stderr)
        for origin, destination, error in failures:
            print(f"  {origin} -> {destination}: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
