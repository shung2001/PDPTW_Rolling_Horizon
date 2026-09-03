import csv
import re
from pathlib import Path


# =========================
# 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    BASE_DIR
    / "자료"
    / "결과"
    / "Ortools"
    / "Rolling_Horizon_구간_60"
    / "d5000"
    / "Penalty_Per_Vehicles"
    / "Time_Solver_60"
    / "130000"
    / "vehicles_150"
    / "node_vehicle_occupancy_by_minute.csv"
)

GROUP_COLUMN = "physical_address"


# 입력 파일 존재 여부 확인
if not INPUT_FILE.is_file():
    raise FileNotFoundError(
        f"입력 CSV 파일을 찾을 수 없습니다:\n{INPUT_FILE}"
    )

OUTPUT_DIR = INPUT_FILE.parent / "분류된_노드"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 파일명에 사용할 수 없는 문자 제거
# =========================

def safe_filename(name):
    return re.sub(r'[\\/:*?"<>|]', "_", str(name)).strip()


# =========================
# CSV 읽기
# =========================

groups = {}

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError("CSV 파일에 열 이름이 없습니다.")

    if GROUP_COLUMN not in fieldnames:
        raise KeyError(
            f"'{GROUP_COLUMN}' 열이 없습니다.\n"
            f"현재 CSV 열: {fieldnames}"
        )

    for row in reader:
        name = row[GROUP_COLUMN]

        if not name:
            name = "주소_없음"

        groups.setdefault(name, []).append(row)


# =========================
# 명칭별 CSV 저장
# =========================

for name, rows in groups.items():

    filename = safe_filename(name) + ".csv"
    output_path = OUTPUT_DIR / filename

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{name}: {len(rows)}행 저장 완료"
        f" → {output_path}"
    )


print(f"\n전체 분류 완료: {len(groups)}개 파일")
print(f"저장 위치: {OUTPUT_DIR}")