"""
Joby eVTOL 5단계 비행 프로파일 기반 A->B 이동시간 추정 모델.

실측 9개 비행(N545JX, 2026.04.24~05.01, 뉴욕) 궤적 분석을 근거로,
급상승-순항-완만하강-급강하-착륙의 5단계로 단순화한 모델이다.

주의: 모든 상수는 "적당히 뭉뚱그린" 근사치이며, 특히 CRUISE_SPEED_KMH(240km/h)는
실측 중앙값(약 176km/h)보다 높게 잡은 사용자 지정값이다. 필요 시 조정할 것.
"""

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 모델 파라미터 (실측 분석 기반 + 일부 사용자 지정 근사치)
# ---------------------------------------------------------------------------

FT_TO_M = 0.3048
KT_TO_MS = 0.514444
KMH_TO_MS = 1 / 3.6

CRUISE_ALT_FT = 800.0
CRUISE_ALT_M = CRUISE_ALT_FT * FT_TO_M          # 243.84 m

CLIMB_ANGLE_DEG = 3.0                            # bin0 실측 평균(2.4~3.3°)

# "고도 0 -> 첫 상승 기록점(150~300ft)" 구간은 지상/호버 위치조정으로 보이는
# 노이즈(속도 0~50kt, 방향 제각각)라 신뢰 가능한 가속도를 뽑을 수 없음.
# 대신 9개 비행 실측 "첫 상승 기록점" 속도(60~105kt, 중앙값 83kt)를
# 급상승 단계의 시작속도로 쓰고, 그 이전 구간(0->83kt 전환)은
# 가속도로 모델링하지 않고 고정 소요시간으로만 반영한다.
CLIMB_V0_KT = 83.0                               # 실측 첫 상승점 속도 중앙값
CLIMB_V0_MS = CLIMB_V0_KT * KT_TO_MS
LIFTOFF_TRANSITION_SEC = 13.0                    # 0->83kt 구간 실측 소요시간(2~13s) 상한값 사용

CRUISE_SPEED_KMH = 240.0                         # 사용자 지정 순항속도
CRUISE_SPEED_MS = CRUISE_SPEED_KMH * KMH_TO_MS

DESC1_ANGLE_DEG = 1.0                            # 완만 하강 (감속 거의 없음)
DESC1_ALT_FRAC = 0.75                            # 전체 고도강하 중 1차가 차지하는 비율

DESC2_ANGLE_DEG = 6.0                            # 근접구역 급강하 (실측 평균 -6.1°)
DESC2_ALT_FRAC = 0.25
LANDING_SPEED_KT = 10.0                          # 착륙 직전 호버 전환 속도 가정
LANDING_SPEED_MS = LANDING_SPEED_KT * KT_TO_MS

# 지상 절차(이륙 전 호버 위치조정, 최종 착지 등)에 들어가는 고정 시간.
# 이륙 전환 구간(LIFTOFF_TRANSITION_SEC)을 여기 포함해 매 계산에 자동 반영한다.
GROUND_PROC_SEC = LIFTOFF_TRANSITION_SEC


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _const_accel_time(d_m, v0_ms, v1_ms):
    """거리 d, 시작속도 v0, 끝속도 v1이 주어졌을 때 등가속 구간의 소요시간(s)과 가속도(m/s^2)."""
    if d_m <= 0:
        return 0.0, 0.0
    avg_v = (v0_ms + v1_ms) / 2.0
    if avg_v <= 0:
        return 0.0, 0.0
    t = d_m / avg_v
    a = (v1_ms - v0_ms) / t if t > 0 else 0.0
    return t, a


@dataclass
class PhaseResult:
    name: str
    distance_m: float
    time_s: float
    v_start_ms: float
    v_end_ms: float
    accel_ms2: float
    angle_deg: float


@dataclass
class TripResult:
    total_distance_m: float
    total_time_s: float
    phases: list
    feasible: bool  # False면 A-B 거리가 너무 짧아 순항 단계까지 못 감(대체 모델 사용)


def estimate_travel_time(lat1, lon1, lat2, lon2) -> TripResult:
    total_d = _haversine_m(lat1, lon1, lat2, lon2)

    # 1) 상승 구간 수평거리 (고도/tan(각도))
    d_climb = CRUISE_ALT_M / math.tan(math.radians(CLIMB_ANGLE_DEG))

    # 2) 하강 구간 (1차: 완만, 2차: 급강하), 고도를 75:25로 배분
    alt_desc1 = CRUISE_ALT_M * DESC1_ALT_FRAC
    alt_desc2 = CRUISE_ALT_M * DESC2_ALT_FRAC
    d_desc1 = alt_desc1 / math.tan(math.radians(DESC1_ANGLE_DEG))
    d_desc2 = alt_desc2 / math.tan(math.radians(DESC2_ANGLE_DEG))

    required_d = d_climb + d_desc1 + d_desc2

    phases = []

    if total_d >= required_d:
        # 정상 케이스: 순항 구간 확보 가능
        feasible = True
        d_cruise = total_d - required_d

        t_climb, a_climb = _const_accel_time(d_climb, CLIMB_V0_MS, CRUISE_SPEED_MS)
        phases.append(PhaseResult("① 급상승", d_climb, t_climb, CLIMB_V0_MS, CRUISE_SPEED_MS, a_climb, CLIMB_ANGLE_DEG))

        t_cruise = d_cruise / CRUISE_SPEED_MS if CRUISE_SPEED_MS > 0 else 0.0
        phases.append(PhaseResult("② 순항", d_cruise, t_cruise, CRUISE_SPEED_MS, CRUISE_SPEED_MS, 0.0, 0.0))

        t_desc1 = d_desc1 / CRUISE_SPEED_MS if CRUISE_SPEED_MS > 0 else 0.0
        phases.append(PhaseResult("③ 1차 하강(완만)", d_desc1, t_desc1, CRUISE_SPEED_MS, CRUISE_SPEED_MS, 0.0, -DESC1_ANGLE_DEG))

        t_desc2, a_desc2 = _const_accel_time(d_desc2, CRUISE_SPEED_MS, LANDING_SPEED_MS)
        phases.append(PhaseResult("④ 2차 하강(급강하)", d_desc2, t_desc2, CRUISE_SPEED_MS, LANDING_SPEED_MS, a_desc2, -DESC2_ANGLE_DEG))

    else:
        # 단거리 케이스: 순항고도/속도까지 못 가고 climb -> 바로 descent로 전환.
        # 상승:하강 수평거리 비율(required_d 기준)로 total_d를 나누고,
        # 도달 가능한 최고속도를 삼각형 프로파일(가속 후 즉시 감속)로 근사.
        feasible = False
        climb_frac = d_climb / required_d
        d_climb_s = total_d * climb_frac
        d_desc_s = total_d - d_climb_s

        # 상승/하강 각도의 가중평균으로 도달 가능한 피크 속도를 등가속-등감속 삼각형으로 계산
        # v_peak^2 = 2*a_climb_est*d_climb_s 형태 대신, 두 구간 거리비로 대칭 피크속도 추정
        avg_accel_est = (CRUISE_SPEED_MS ** 2 - CLIMB_V0_MS ** 2) / (2 * d_climb) if d_climb > 0 else 0
        v_peak = math.sqrt(max(CLIMB_V0_MS ** 2 + 2 * avg_accel_est * d_climb_s, 0))
        v_peak = min(v_peak, CRUISE_SPEED_MS)

        t_climb_s, a1 = _const_accel_time(d_climb_s, CLIMB_V0_MS, v_peak)
        phases.append(PhaseResult("① 급상승(순항 미도달)", d_climb_s, t_climb_s, CLIMB_V0_MS, v_peak, a1, CLIMB_ANGLE_DEG))

        t_desc_s, a2 = _const_accel_time(d_desc_s, v_peak, LANDING_SPEED_MS)
        phases.append(PhaseResult("④ 급강하(순항 미도달)", d_desc_s, t_desc_s, v_peak, LANDING_SPEED_MS, a2, -DESC2_ANGLE_DEG))

    total_time = sum(p.time_s for p in phases) + GROUND_PROC_SEC

    return TripResult(total_distance_m=total_d, total_time_s=total_time, phases=phases, feasible=feasible)


def _fmt(v):
    return f"{v:.1f}"


if __name__ == "__main__":
    # 예시: 뉴욕 다운타운 헬리포트 -> JFK 인근 (실측 데이터의 대표 좌표)
    examples = [
        ("다운타운 -> JFK 근처", 40.701214, -74.008965, 40.652847, -73.822571),
        ("단거리 셔틀 예시", 40.697891, -74.011414, 40.700157, -73.999535),
    ]

    for label, lat1, lon1, lat2, lon2 in examples:
        r = estimate_travel_time(lat1, lon1, lat2, lon2)
        print(f"\n=== {label} ===")
        print(f"총 거리: {r.total_distance_m/1000:.2f} km, 총 소요시간: {r.total_time_s/60:.2f} 분 "
              f"(순항 도달: {'예' if r.feasible else '아니오 - 단거리 삼각형 프로파일'})")
        print(f"  0 이륙 전환(고정)      -                   시간 {GROUND_PROC_SEC:6.1f}s  "
              f"속도   0.0-> {CLIMB_V0_MS*3.6:5.1f} km/h  가속도  (미측정, 노이즈)  각도  -")
        for p in r.phases:
            print(f"  {p.name:22s} 거리 {p.distance_m/1000:6.2f}km  시간 {p.time_s:6.1f}s  "
                  f"속도 {p.v_start_ms*3.6:5.1f}->{p.v_end_ms*3.6:5.1f} km/h  가속도 {p.accel_ms2:+.2f} m/s²  각도 {p.angle_deg:+.1f}°")
