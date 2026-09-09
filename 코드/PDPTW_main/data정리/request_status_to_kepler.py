"""요청 상태 CSV에 노드 좌표를 연결해 kepler.gl용 CSV를 생성한다.

실행: python request_status_to_kepler.py
다른 파일: python request_status_to_kepler.py --input INPUT.csv --output OUTPUT.csv

추가 패키지가 필요 없다. 기본 출력은 입력 폴더의
rolling_horizon_request_status_kepler.csv이다. 원본 열과 모든 상태를 보존한다.
kepler.gl에서 출력 CSV를 업로드하면 origin/destination Point 및 Arc 레이어를
인식한다. Arc의 출발 좌표는 origin_lat/origin_lng, 도착 좌표는
destination_lat/destination_lng이다. final_status로 색상이나 필터를 설정한다.
시간 필터는 기존 time_window_start, pickup_time 등의 분 단위 숫자 열을 사용한다.
시뮬레이션의 0분은 05:40이다. 실제 날짜는 원본에 없어 생성하지 않는다.
Arc는 요청의 OD 연결이며 실제 차량 이동 경로를 나타내지 않는다.

형식 참고: https://docs.kepler.gl/docs/user-guides/b-kepler-gl-workflow/a-add-data-to-the-map
"""

from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path


PDPTW_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    PDPTW_ROOT / "자료" / "결과" / "Ortools" / "Rolling_Horizon_구간_30"
    / "d5000" / "Penalty_Per_Vehicles" / "Time_Solver_120"
    / "add_6mins_penalty_수정_2" / "Original" / "rolling_horizon_request_status.csv"
)
DEFAULT_NODES = PDPTW_ROOT / "자료" / "기초자료" / "vp_reference.csv"
ADDED_COLUMNS = [
    "origin_name", "origin_lat", "origin_lng",
    "destination_name", "destination_lat", "destination_lng",
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            content = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"UTF-8 또는 CP949로 읽을 수 없는 파일: {path}")
    reader = csv.DictReader(io.StringIO(content, newline=""))
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)):
        raise ValueError(f"헤더가 없거나 중복된 열 이름이 있습니다: {path}")
    rows = list(reader)
    for line, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path}: {line}행의 열 수가 헤더와 다릅니다.")
    return fields, rows


def node_id(value: str) -> str:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"노드 번호는 정수여야 합니다: {value!r}")
    return str(int(number))


def convert(input_path: Path, nodes_path: Path, output_path: Path) -> int:
    if output_path.resolve() in (input_path.resolve(), nodes_path.resolve()):
        raise ValueError("출력 경로는 입력 파일 및 노드 파일과 달라야 합니다.")
    fields, requests = read_csv(input_path)
    node_fields, nodes = read_csv(nodes_path)
    for required, available, path in (
        ({"origin", "destination"}, fields, input_path),
        ({"vp_id", "vpname", "lat", "lon"}, node_fields, nodes_path),
    ):
        missing = required - set(available)
        if missing:
            raise ValueError(f"{path}: 필수 열 누락 {sorted(missing)}")
    if set(fields) & set(ADDED_COLUMNS):
        raise ValueError("입력 파일에 변환용 열이 이미 존재합니다. 원본 CSV를 지정하세요.")
    lookup = {}
    for row in nodes:
        key = node_id(row["vp_id"])
        if key in lookup:
            raise ValueError(f"노드 참조 파일의 중복 vp_id: {key}")
        lat, lng = float(row["lat"]), float(row["lon"])
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            raise ValueError(f"노드 {key}: 위·경도 범위 오류")
        lookup[key] = (row["vpname"], lat, lng)
    for line, row in enumerate(requests, 2):
        for endpoint in ("origin", "destination"):
            key = node_id(row[endpoint])
            if key not in lookup:
                raise ValueError(f"입력 {line}행: {endpoint} 노드 {key}의 좌표가 없습니다.")
            name, lat, lng = lookup[key]
            row.update({f"{endpoint}_name": name, f"{endpoint}_lat": lat,
                        f"{endpoint}_lng": lng})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + ADDED_COLUMNS)
        writer.writeheader()
        writer.writerows(requests)
    return len(requests)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="요청 상태 CSV")
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES, help="노드 좌표 CSV")
    parser.add_argument("--output", type=Path, help="출력 CSV (생략 시 입력 폴더의 *_kepler.csv)")
    args = parser.parse_args()
    output = args.output or args.input.with_name(f"{args.input.stem}_kepler.csv")
    try:
        count = convert(args.input, args.nodes, output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"변환 실패: {exc}\n")
    print(f"변환 완료: {count:,}개 요청\n저장: {output.resolve()}")


if __name__ == "__main__":
    main()
