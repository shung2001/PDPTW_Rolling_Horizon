"""m과 n이 모두 참조 vp_id에 포함된 수요 행만 저장한다.

실행: python PDPTW/코드/filter_demand_by_vp.py
다른 파일 처리: --reference 경로 --input 경로 --output 경로
"""

import argparse
import csv
import io
from decimal import Decimal, InvalidOperation
from pathlib import Path


def read_csv(path):
    """UTF-8 및 한국어 Windows CSV(CP949)를 읽는다."""
    data = path.read_bytes()
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = data.decode("cp949")
    return io.StringIO(content, newline="")


def numeric_id(value):
    """2와 2.0 등을 동일한 정수 ID로 해석한다."""
    try:
        number = Decimal(value.strip())
    except (InvalidOperation, AttributeError):
        return None
    if not number.is_finite() or number != number.to_integral_value():
        return None
    return int(number)


def main():
    base = Path(__file__).resolve().parents[1] / "자료" / "기초자료"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=base / "vp_reference_축소.csv")
    parser.add_argument("--input", type=Path, default=base / "finalDemand_v5" / "finalDemand_v5" / "d5000_s01.csv")
    parser.add_argument("--output", type=Path, default=base / "finalDemand_v5" / "축소" / "d5000_s01.csv")
    args = parser.parse_args()
    if args.output.resolve() in {args.input.resolve(), args.reference.resolve()}:
        parser.error("출력 경로는 원본 및 참조 파일과 달라야 합니다.")

    with read_csv(args.reference) as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "vp_id" not in reader.fieldnames:
            raise ValueError("참조 파일에 vp_id 열이 없습니다.")
        valid_ids = {numeric_id(row["vp_id"]) for row in reader}
        valid_ids.discard(None)
    if not valid_ids:
        raise ValueError("참조 파일에 유효한 vp_id가 없습니다.")

    with read_csv(args.input) as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        if not fields or not {"m", "n"}.issubset(fields):
            raise ValueError("수요 파일에 m, n 열이 필요합니다.")
        rows = list(reader)
    kept = [row for row in rows if numeric_id(row["m"]) in valid_ids and numeric_id(row["n"]) in valid_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    print(f"vp_id: {sorted(valid_ids)}")
    print(f"원본 {len(rows):,}행 / 유지 {len(kept):,}행 / 제거 {len(rows) - len(kept):,}행")
    print(f"저장: {args.output}")


if __name__ == "__main__":
    main()
