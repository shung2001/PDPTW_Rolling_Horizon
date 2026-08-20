# -*- coding: utf-8 -*-
"""
10개 Depot Excel 파일의 시간대별 vehicle_count를 읽어
시간(x축) × Depot(y축) 스펙트럼(Heatmap)을 출력한다.

필수 열:
- time_min
- vehicle_count

BASE_TIME = 05:40 기준으로 time_min을 실제 HH:MM 시각으로 변환한다.
"""

from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. 사용자 설정
# ============================================================

# Excel 파일들이 들어 있는 폴더
DATA_DIR = Path(
    r"C:\Users\choih\Desktop\연구\PDPTW\자료\결과\Ortools\S1_1\Penalty_per_vehicles\Remove_depot\vehicles_55\depot"
)

# 실제 파일명이 다르면 오른쪽 파일명만 수정하면 된다.
DEPOT_FILES = {
    "서울역": "서울역.xlsx",
    "수서": "수서.xlsx",
    "삼성": "삼성.xlsx",
    "여의도": "여의도.xlsx",
    "김포공항": "김포공항.xlsx",
    "일산": "일산.xlsx",
    "판교": "판교.xlsx",
    "동탄/용인": "동탄,용인.xlsx",
    "평택": "평택.xlsx",
    "인천공항": "인천공항.xlsx",
}

# PDPTW 코드의 BASE_TIME
BASE_TIME = "05:40"

# x축 시간 라벨 간격(분)
TIME_TICK_INTERVAL_MIN = 120

# 그래프 저장 여부
SAVE_FIGURE = True

# 저장 파일명
OUTPUT_PATH = DATA_DIR / "depot_vehicle_count_spectrum.png"


# ============================================================
# 2. 한글 폰트 설정
# ============================================================

# Windows 기본 한글 폰트
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3. Excel 자료 읽기
# ============================================================

depot_series = {}
all_times = set()

for depot_name, file_name in DEPOT_FILES.items():
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다.\n"
            f"Depot: {depot_name}\n"
            f"경로: {file_path}"
        )

    df = pd.read_excel(file_path)

    required_columns = {"time_min", "vehicle_count"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise KeyError(
            f"{file_name}에 필요한 열이 없습니다: {sorted(missing_columns)}\n"
            f"실제 열: {list(df.columns)}"
        )

    # 필요한 열만 사용
    temp = df[["time_min", "vehicle_count"]].copy()

    temp["time_min"] = pd.to_numeric(
        temp["time_min"], errors="coerce"
    )
    temp["vehicle_count"] = pd.to_numeric(
        temp["vehicle_count"], errors="coerce"
    )

    temp = temp.dropna(subset=["time_min", "vehicle_count"])

    temp["time_min"] = temp["time_min"].astype(int)
    temp["vehicle_count"] = temp["vehicle_count"].astype(int)

    # 동일한 time_min이 여러 개 존재할 경우 합계
    # 현재 자료가 시간당 한 행이라면 값은 그대로 유지된다.
    series = temp.groupby("time_min")["vehicle_count"].sum()

    depot_series[depot_name] = series
    all_times.update(series.index.tolist())


if not all_times:
    raise ValueError("유효한 time_min 데이터가 없습니다.")


# ============================================================
# 4. 모든 Depot의 시간축 통일
# ============================================================

min_time = min(all_times)
max_time = max(all_times)

# 분 단위 전체 시간축
time_axis = np.arange(min_time, max_time + 1)

# 행 = Depot
# 열 = time_min
matrix = np.zeros(
    (len(DEPOT_FILES), len(time_axis)),
    dtype=float,
)

for row_index, depot_name in enumerate(DEPOT_FILES.keys()):
    series = depot_series[depot_name]

    # 데이터가 없는 시간은 0대
    aligned = series.reindex(
        time_axis,
        fill_value=0
    )

    matrix[row_index, :] = aligned.to_numpy()


# ============================================================
# 5. BASE_TIME 기준 실제 시각 생성
# ============================================================

base_hour, base_minute = map(
    int,
    BASE_TIME.split(":")
)

base_datetime = datetime(
    2000,
    1,
    1,
    base_hour,
    base_minute
)

tick_times = np.arange(
    min_time,
    max_time + 1,
    TIME_TICK_INTERVAL_MIN
)

# imshow의 x좌표는 0부터 시작하므로 보정
tick_positions = tick_times - min_time

tick_labels = [
    (
        base_datetime
        + timedelta(minutes=int(t))
    ).strftime("%H:%M")
    for t in tick_times
]


# ============================================================
# 6. 시간대별 차량 대수 스펙트럼 출력
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 7)
)

image = ax.imshow(
    matrix,
    aspect="auto",
    interpolation="nearest",
    origin="upper",
)

# y축: Depot
ax.set_yticks(
    np.arange(len(DEPOT_FILES))
)

ax.set_yticklabels(
    list(DEPOT_FILES.keys())
)

# x축: 실제 시간
ax.set_xticks(
    tick_positions
)

ax.set_xticklabels(
    tick_labels
)

ax.set_xlabel("시간")
ax.set_ylabel("Depot")

ax.set_title(
    "시간대별 Depot 차량 대수 스펙트럼"
)

# 차량 대수 Color Bar
colorbar = fig.colorbar(
    image,
    ax=ax
)

colorbar.set_label(
    "차량 대수"
)

fig.tight_layout()


# ============================================================
# 7. 그래프 저장
# ============================================================

if SAVE_FIGURE:
    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"그래프 저장 완료: {OUTPUT_PATH}"
    )


# ============================================================
# 8. Depot별 최대 차량 대수 및 발생 시각 출력
# ============================================================

print("\n[Depot별 최대 vehicle_count]")

for row_index, depot_name in enumerate(DEPOT_FILES.keys()):
    row = matrix[row_index]

    max_count = int(row.max())
    max_index = int(row.argmax())

    peak_time_min = int(
        time_axis[max_index]
    )

    peak_time_hhmm = (
        base_datetime
        + timedelta(minutes=peak_time_min)
    ).strftime("%H:%M")

    print(
        f"{depot_name:8s} : "
        f"최대 {max_count:2d}대 "
        f"({peak_time_hhmm})"
    )


plt.show()
