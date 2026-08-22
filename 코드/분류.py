import csv
import re
from pathlib import Path

# =========================
# 설정
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "자료" / "결과" / "Ortools" / "8월21일" / "FOCUSED" / "Remove_Depot" / "node_vehicle_occupancy_by_minute.csv"
OUTPUT_DIR = INPUT_FILE.parent / "분류된_노드"

# 명칭이 들어있는 열
GROUP_COLUMN = "physical_address"

OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# 파일명에 사용할 수 없는 문자 제거
# =========================
def safe_filename(name):
    return re.sub(r'[\\/:*?"<>|]', "_", str(name))


# =========================
# CSV 읽기
# =========================
groups = {}

with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    fieldnames = reader.fieldnames

    for row in reader:
        name = row[GROUP_COLUMN]

        if name not in groups:
            groups[name] = []

        groups[name].append(row)


# =========================
# 명칭별 CSV 저장
# =========================
for name, rows in groups.items():

    filename = safe_filename(name) + ".csv"
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)

    print(f"{name}: {len(rows)}행 저장 완료 → {output_path}")


print("\n전체 분류 완료")