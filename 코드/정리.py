# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


# ============================================================
# 경로 설정
# ============================================================

BASE_DIR = Path(
    r"C:\Users\c\Desktop\대학생활\학연생\새로운_경로방식"
    r"\PDPTW_Rolling_Horizon\자료\결과\Ortools\S1_1"
    r"\Penalty_per_vehicles\Remove_Pending_Depot"
)

OUTPUT_PATH = BASE_DIR / "penalty_per_vehicles.csv"


# ============================================================
# 차량 수별 종합 Penalty 계산
# ============================================================

results = []

# vehicles_50 ~ vehicles_200
for vehicle_count in range(50, 201):

    vehicle_dir = BASE_DIR / f"vehicles_{vehicle_count}"

    status_path = vehicle_dir / "rolling_horizon_request_status.csv"

    # 해당 vehicle 폴더가 없으면 건너뜀
    if not vehicle_dir.exists():
        continue

    # 결과 CSV가 없으면 건너뜀
    if not status_path.exists():
        print(
            f"[WARNING] 차량 {vehicle_count}대: "
            f"rolling_horizon_request_status.csv 없음"
        )
        continue

    try:
        df = pd.read_csv(status_path)

        # 필요한 열 확인
        required_columns = {
            "final_status",
            "final_drop_penalty",
        }

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            print(
                f"[WARNING] 차량 {vehicle_count}대: "
                f"필수 열 없음 -> {missing_columns}"
            )
            continue

        # ----------------------------------------------------
        # 최종 Overdue 요청만 추출
        # ----------------------------------------------------
        overdue_df = df[
            df["final_status"].astype(str).str.strip() == "Overdue"
        ]

        complete_df = df[
            df["final_status"].astype(str).str.strip() == "Complete"
        ]

        # ----------------------------------------------------
        # 종합 Penalty
        # = Overdue의 final_drop_penalty 합계
        # ----------------------------------------------------
        total_penalty = pd.to_numeric(
            overdue_df["final_drop_penalty"],
            errors="coerce",
        ).fillna(0).sum()

        complete_count = len(complete_df)
        overdue_count = len(overdue_df)
        total_count = len(df)

        results.append(
            {
                "vehicle_count": vehicle_count,
                "total_penalty": int(total_penalty),
                "complete_count": complete_count,
                "overdue_count": overdue_count,
                "total_batch_count": total_count,
            }
        )

        print(
            f"차량 {vehicle_count:>3}대 | "
            f"Penalty={int(total_penalty):,} | "
            f"Complete={complete_count:,} | "
            f"Overdue={overdue_count:,}"
        )

    except Exception as e:
        print(
            f"[ERROR] 차량 {vehicle_count}대 처리 실패: {e}"
        )


# ============================================================
# 결과 저장
# ============================================================

if not results:
    raise RuntimeError(
        "계산 가능한 vehicles_* 결과가 없습니다."
    )

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    "vehicle_count"
).reset_index(drop=True)

result_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


print()
print("=" * 70)
print("종합 Penalty 계산 완료")
print(f"총 Vehicle 조건 수 : {len(result_df)}")
print(f"저장 위치          : {OUTPUT_PATH}")
print("=" * 70)