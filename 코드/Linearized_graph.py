# -*- coding: utf-8 -*-
"""
차량 대수에 따른 Penalty의 지수 비선형 회귀를 수행하고,

1. SciPy optimizer 수렴 여부
2. R² / RMSE / NRMSE
3. 회귀곡선의 95% confidence band
4. Practical convergence (90%, 95%, 99%)
5. Geometric knee point
6. Residual diagnostic
7. 지수함수 선형화 diagnostic

을 계산하고 시각화한다.

회귀모델:
    P(V) = C + D * exp[-k * (V - V0)]

여기서
    V  : 차량 수
    V0 : 실제 분석 데이터의 기준 차량 수
    C  : 차량 수가 충분히 증가할 때 접근하는 점근 Penalty
    D  : V0에서 점근선 C까지 남아 있는 Penalty 감소 가능량
    k  : 차량 증가에 따른 Penalty 감소율
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit
from scipy.stats import t


# ============================================================
# 0. 경로 및 분석 설정
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = (
    BASE_DIR.parent
    / "자료"
    / "결과"
    / "Ortools"
    / "8월21일"
    / "Penalty_per_vehicles"
    / "Remove_depot"
    / "penalty_per_vehicles.csv"
)

SAVE_PATH = (
    BASE_DIR.parent
        / "자료"
        / "결과"
        / "Ortools"
        / "8월21일"
        / "Penalty_per_vehicles"
        / "Remove_depot"
)
# ------------------------------------------------------------
# SciPy optimizer의 numerical stopping tolerance
#
# 이것은 "Penalty가 수렴했다고 판단하는 기준"이 아니라
# curve_fit 내부 최적화 알고리즘의 반복 종료 기준이다.
# ------------------------------------------------------------
OPTIMIZER_TOLERANCE = 1e-8


# 통계적 신뢰수준
CONFIDENCE_LEVEL = 0.95


# ------------------------------------------------------------
# Practical convergence 기준
#
# V0에서 점근선 C까지 남아 있던 Penalty 감소 가능량 중
# 몇 %가 감소했는가를 기준으로 한다.
#
# 예:
# 0.95 -> 초기 감소 가능량의 95%가 감소한 시점
# ------------------------------------------------------------
PRIMARY_CONVERGENCE_RATE = 0.95

# 95%라는 기준의 임의성을 확인하기 위한 sensitivity analysis
SENSITIVITY_CONVERGENCE_RATES = (
    0.90,
    0.95,
    0.99,
)


# ------------------------------------------------------------
# 회귀에서 제외할 비정상 구간
#
# 반드시 Solver 문제, local optimum, 비정상 실행 등
# 연구적으로 설명 가능한 근거가 있을 때만 제외할 것.
#
# False : exclude mask를 사용하지 않고 모든 데이터를 회귀에 사용
# True  : EXCLUDED_RANGES에 지정한 구간을 회귀에서 제외
# ------------------------------------------------------------
USE_EXCLUSION_MASK = False

EXCLUDED_RANGES = [
    (147, 164),
]


# 지수 감쇠계수 k의 초기 추정값
INITIAL_K = 0.02


# Knee point 계산용 고해상도 grid
KNEE_GRID_SIZE = 5000


# 그래프용 grid
CURVE_GRID_SIZE = 1000


# 외삽 때문에 그래프가 지나치게 늘어나는 것을 방지
MAX_PLOT_EXTRAPOLATION_FACTOR = 2.0


# 추가 diagnostic plot 출력 여부
SHOW_RESIDUAL_PLOT = True
SHOW_LINEARIZED_DIAGNOSTIC = True


# ============================================================
# 1. 함수 정의
# ============================================================

def exponential_model(vehicle_count, C, D, k, reference_vehicle):
    """
    기준 차량 V0를 이용한 지수감쇠 모델.

    P(V) = C + D * exp[-k(V - V0)]
    """

    vehicle_count = np.asarray(vehicle_count, dtype=float)

    return (
        C
        + D
        * np.exp(
            -k * (vehicle_count - reference_vehicle)
        )
    )


def prediction_standard_error(
    vehicle_count,
    params,
    covariance,
    reference_vehicle,
):
    """
    Delta method를 이용하여
    fitted mean curve의 표준오차를 계산한다.

    주의:
    이것은 새로운 개별 관측값의 prediction interval이 아니라
    추정된 평균 회귀곡선의 confidence interval을 위한 것이다.
    """

    C, D, k = params

    vehicle_count = np.asarray(
        vehicle_count,
        dtype=float,
    )

    delta_vehicle = (
        vehicle_count - reference_vehicle
    )

    exp_term = np.exp(
        -k * delta_vehicle
    )


    # --------------------------------------------------------
    # Jacobian
    #
    # P(V) = C + D exp[-k(V-V0)]
    #
    # ∂P/∂C = 1
    # ∂P/∂D = exp[-k(V-V0)]
    # ∂P/∂k = -D(V-V0)exp[-k(V-V0)]
    # --------------------------------------------------------

    jacobian = np.column_stack(
        (
            np.ones_like(vehicle_count),
            exp_term,
            -D
            * delta_vehicle
            * exp_term,
        )
    )


    variance = np.einsum(
        "ij,jk,ik->i",
        jacobian,
        covariance,
        jacobian,
    )


    return np.sqrt(
        np.maximum(
            variance,
            0.0,
        )
    )


def goodness_of_fit(
    y_true,
    y_pred,
):
    """
    회귀 적합도를 계산한다.

    반환값:
        R²
        RMSE
        NRMSE
        residual
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )


    residual = (
        y_true - y_pred
    )


    # Sum of Squared Errors
    sse = float(
        np.sum(
            residual ** 2
        )
    )


    # Total Sum of Squares
    sst = float(
        np.sum(
            (
                y_true
                - np.mean(y_true)
            ) ** 2
        )
    )


    if sst == 0.0:
        r_squared = np.nan
    else:
        r_squared = (
            1.0
            - sse / sst
        )


    rmse = float(
        np.sqrt(
            np.mean(
                residual ** 2
            )
        )
    )


    data_range = float(
        np.max(y_true)
        - np.min(y_true)
    )


    if data_range == 0.0:
        nrmse = np.nan
    else:
        nrmse = (
            rmse / data_range
        )


    return (
        r_squared,
        rmse,
        nrmse,
        residual,
    )


def practical_convergence_vehicle(
    convergence_rate,
    k,
    reference_vehicle,
):
    """
    Practical convergence 차량 수를 계산한다.

    기준:

        P(V) - C
        ---------------- = 1 - convergence_rate
        P(V0) - C


    지수모델에서는

        exp[-k(V-V0)]
        = 1 - convergence_rate


    따라서

        V =
        V0 - ln(1-r) / k
    """

    remaining_rate = (
        1.0
        - convergence_rate
    )


    if not (
        0.0
        < remaining_rate
        < 1.0
    ):
        raise ValueError(
            "convergence_rate는 "
            "0과 1 사이여야 합니다."
        )


    return (
        reference_vehicle
        - np.log(
            remaining_rate
        )
        / k
    )


def geometric_knee_point(
    x_grid,
    y_grid,
):
    """
    정규화된 회귀곡선에서
    시작점과 끝점을 잇는 직선(chord)과의 거리가
    최대가 되는 점을 geometric knee로 정의한다.

    x, y를 각각 [0, 1]로 정규화하기 때문에
    Penalty 단위가 매우 큰 것의 영향을 줄일 수 있다.

    이 값은 '수학적 최적점'이라기보다
    한계효용이 크게 둔화되는 지점을 찾기 위한
    heuristic indicator이다.
    """

    x_grid = np.asarray(
        x_grid,
        dtype=float,
    )

    y_grid = np.asarray(
        y_grid,
        dtype=float,
    )


    x_range = (
        x_grid.max()
        - x_grid.min()
    )

    y_range = (
        y_grid.max()
        - y_grid.min()
    )


    if (
        x_range <= 0.0
        or y_range <= 0.0
    ):
        raise ValueError(
            "Knee point 계산에 필요한 "
            "데이터 변화 범위가 없습니다."
        )


    # 0~1 정규화
    x_norm = (
        x_grid - x_grid.min()
    ) / x_range

    y_norm = (
        y_grid - y_grid.min()
    ) / y_range


    start_point = np.array(
        [
            x_norm[0],
            y_norm[0],
        ]
    )

    end_point = np.array(
        [
            x_norm[-1],
            y_norm[-1],
        ]
    )


    line_vector = (
        end_point
        - start_point
    )

    line_length = np.linalg.norm(
        line_vector
    )


    if line_length == 0.0:
        raise ValueError(
            "Knee point 계산용 "
            "직선 길이가 0입니다."
        )


    points = np.column_stack(
        (
            x_norm,
            y_norm,
        )
    )


    point_vectors = (
        points
        - start_point
    )


    # 각 점과 chord 사이의 수직거리
    distances = np.abs(
        line_vector[0]
        * point_vectors[:, 1]
        -
        line_vector[1]
        * point_vectors[:, 0]
    ) / line_length


    knee_index = int(
        np.argmax(
            distances
        )
    )


    return (
        float(
            x_grid[knee_index]
        ),
        float(
            y_grid[knee_index]
        ),
        float(
            distances[knee_index]
        ),
    )


def one_vehicle_penalty_reduction(
    vehicle_count,
    D,
    k,
    reference_vehicle,
):
    """
    차량을 정확히 한 대 추가했을 때
    회귀모델이 예측하는 Penalty 감소량.

        P(V) - P(V+1)
    """

    remaining_penalty = (
        D
        * np.exp(
            -k
            * (
                vehicle_count
                - reference_vehicle
            )
        )
    )


    reduction = (
        remaining_penalty
        * (
            1.0
            - np.exp(-k)
        )
    )


    return float(
        reduction
    )


def make_exclusion_mask(
    vehicle_values,
    excluded_ranges,
):
    """
    제외 범위에 해당하지 않는 데이터만
    True인 mask를 만든다.
    """

    mask = np.ones_like(
        vehicle_values,
        dtype=bool,
    )


    for lower, upper in excluded_ranges:

        mask &= ~(
            (
                vehicle_values >= lower
            )
            &
            (
                vehicle_values <= upper
            )
        )


    return mask


# ============================================================
# 2. CSV 데이터 로드
# ============================================================

df = pd.read_csv(
    CSV_PATH
)


required_columns = {
    "vehicle_count",
    "total_penalty",
}


missing_columns = (
    required_columns
    - set(df.columns)
)


if missing_columns:

    raise KeyError(
        "CSV에 필요한 열이 없습니다: "
        f"{sorted(missing_columns)}"
    )


x = pd.to_numeric(
    df["vehicle_count"],
    errors="raise",
).to_numpy(
    dtype=float
)


y = pd.to_numeric(
    df["total_penalty"],
    errors="raise",
).to_numpy(
    dtype=float
)


# NaN / infinity 검사
finite_mask = (
    np.isfinite(x)
    & np.isfinite(y)
)


if not np.all(
    finite_mask
):

    raise ValueError(
        "num_vehicles 또는 total_penalty에 "
        "NaN / inf 값이 포함되어 있습니다."
    )


if len(x) < 4:

    raise ValueError(
        "3개 파라미터를 추정하기 위해서는 "
        "최소 4개 이상의 데이터가 필요합니다."
    )


# ============================================================
# 3. 비정상 구간 제외
# ============================================================

if USE_EXCLUSION_MASK:
    fit_mask = make_exclusion_mask(
        x,
        EXCLUDED_RANGES,
    )
else:
    # exclude 기능은 보존하되 현재 분석에서는 모든 데이터를 사용한다.
    fit_mask = np.ones_like(
        x,
        dtype=bool,
    )


x_fit = x[
    fit_mask
]

y_fit = y[
    fit_mask
]


x_excluded = x[
    ~fit_mask
]

y_excluded = y[
    ~fit_mask
]


if len(x_fit) <= 3:

    raise ValueError(
        "제외 후 데이터가 너무 적습니다."
    )


# ============================================================
# 4. 기준 차량 V0 설정
# ============================================================

REFERENCE_VEHICLE = float(
    np.min(
        x_fit
    )
)


reference_values = y_fit[
    np.isclose(
        x_fit,
        REFERENCE_VEHICLE,
    )
]


REFERENCE_PENALTY = float(
    np.mean(
        reference_values
    )
)


# ============================================================
# 5. curve_fit용 모델 정의
# ============================================================

def fit_model(
    vehicle_count,
    C,
    D,
    k,
):

    return exponential_model(
        vehicle_count,
        C,
        D,
        k,
        REFERENCE_VEHICLE,
    )


# ============================================================
# 6. 초기 파라미터 설정
# ============================================================

minimum_penalty = float(
    np.min(
        y_fit
    )
)


# 점근선 C 초기값:
# 관측 최솟값보다 약간 낮은 값에서 시작
C0 = max(
    0.0,
    minimum_penalty * 0.90,
)


# V0에서 점근선까지의 차이
D0 = max(
    REFERENCE_PENALTY - C0,
    1.0,
)


k0 = INITIAL_K


initial_guess = [
    C0,
    D0,
    k0,
]


# ============================================================
# 7. Parameter bounds
# ============================================================

# 물리적으로
#
# C >= 0
# D >= 0
# k > 0
#
# 를 강제한다.

lower_bounds = [
    0.0,
    0.0,
    1e-12,
]


upper_bounds = [
    np.inf,
    np.inf,
    np.inf,
]


# ============================================================
# 8. 비선형 최소제곱 회귀
# ============================================================

params, covariance, info, message, ier = curve_fit(

    fit_model,

    x_fit,
    y_fit,

    p0=initial_guess,

    bounds=(
        lower_bounds,
        upper_bounds,
    ),

    # bounds를 사용하므로 TRF 방법을 명시적으로 사용
    method="trf",

    # SciPy numerical stopping criteria
    ftol=OPTIMIZER_TOLERANCE,
    xtol=OPTIMIZER_TOLERANCE,
    gtol=OPTIMIZER_TOLERANCE,

    # C, D와 k의 크기 차이를 완화하기 위한 scaling
    x_scale="jac",

    # TRF는 maxfev가 아니라 max_nfev 사용
    max_nfev=100000,

    full_output=True,
)


C, D, k = params


# ============================================================
# 9. SciPy optimizer 수렴 확인
# ============================================================

optimizer_converged = (
    ier in (
        1,
        2,
        3,
        4,
    )
)


if not optimizer_converged:

    raise RuntimeError(
        "회귀 최적화가 수렴하지 않았습니다.\n"
        f"ier = {ier}\n"
        f"message = {message}"
    )


if k <= 0.0:

    raise ValueError(
        "k가 양수가 아니므로 "
        "감소형 지수모델이 성립하지 않습니다.\n"
        f"k = {k}"
    )


# ============================================================
# 10. 기존 식 P=C+A exp(-kV)의 A로 환산
# ============================================================

# D exp[-k(V-V0)]
#
# = D exp(kV0) exp(-kV)
#
# 따라서
#
# A = D exp(kV0)

A_equivalent = (
    D
    * np.exp(
        k
        * REFERENCE_VEHICLE
    )
)


# ============================================================
# 11. 회귀 적합도
# ============================================================

y_fit_pred = fit_model(
    x_fit,
    *params,
)


(
    r_squared,
    rmse,
    nrmse,
    residuals,
) = goodness_of_fit(
    y_fit,
    y_fit_pred,
)


# 비정상 구간을 포함한 전체 데이터에 대한
# 참고용 지표도 계산
y_all_pred = fit_model(
    x,
    *params,
)


(
    r_squared_all,
    rmse_all,
    nrmse_all,
    _,
) = goodness_of_fit(
    y,
    y_all_pred,
)


# ============================================================
# 12. 통계적 신뢰구간 설정
# ============================================================

n_observations = len(
    y_fit
)

n_parameters = len(
    params
)


degrees_of_freedom = (
    n_observations
    - n_parameters
)


if degrees_of_freedom <= 0:

    raise ValueError(
        "신뢰구간 계산을 위한 "
        "자유도가 부족합니다."
    )


alpha = (
    1.0
    - CONFIDENCE_LEVEL
)


# 단순히 1.96을 고정하지 않고
# 실제 자유도에 따라 Student-t 임계값 사용
critical_value = float(
    t.ppf(
        1.0 - alpha / 2.0,
        degrees_of_freedom,
    )
)


# ============================================================
# 13. Parameter confidence intervals
# ============================================================

parameter_standard_errors = np.sqrt(
    np.maximum(
        np.diag(
            covariance
        ),
        0.0,
    )
)


parameter_ci = np.column_stack(
    (
        params
        - critical_value
        * parameter_standard_errors,

        params
        + critical_value
        * parameter_standard_errors,
    )
)


# ============================================================
# 14. Practical convergence 계산
# ============================================================

primary_convergence_vehicle = (
    practical_convergence_vehicle(
        PRIMARY_CONVERGENCE_RATE,
        k,
        REFERENCE_VEHICLE,
    )
)


primary_convergence_vehicle_integer = int(
    np.ceil(
        primary_convergence_vehicle
    )
)


remaining_rate = (
    1.0
    - PRIMARY_CONVERGENCE_RATE
)


# V95에서는
#
# P(V95)
# = C + D * 0.05
#
# 가 된다.
primary_convergence_penalty = float(
    C
    + D
    * remaining_rate
)


# ============================================================
# 15. Practical convergence 차량 수의 신뢰구간
# ============================================================

# V_r =
# V0 - ln(1-r)/k
#
# 따라서
#
# dV_r/dk =
# ln(1-r)/k²

k_variance = max(
    float(
        covariance[2, 2]
    ),
    0.0,
)


vehicle_derivative_k = (
    np.log(
        remaining_rate
    )
    / (k ** 2)
)


vehicle_standard_error = (
    abs(
        vehicle_derivative_k
    )
    * np.sqrt(
        k_variance
    )
)


vehicle_ci = (
    primary_convergence_vehicle
    - critical_value
    * vehicle_standard_error,

    primary_convergence_vehicle
    + critical_value
    * vehicle_standard_error,
)


# ============================================================
# 16. Practical convergence Penalty 신뢰구간
# ============================================================

# P(V_r)
# = C + D(1-r)
#
# Gradient =
#
# [1, 1-r, 0]

penalty_gradient = np.array(
    [
        1.0,
        remaining_rate,
        0.0,
    ]
)


penalty_variance = max(
    float(
        penalty_gradient
        @ covariance
        @ penalty_gradient
    ),
    0.0,
)


penalty_standard_error = np.sqrt(
    penalty_variance
)


penalty_ci = (
    primary_convergence_penalty
    - critical_value
    * penalty_standard_error,

    primary_convergence_penalty
    + critical_value
    * penalty_standard_error,
)


# ============================================================
# 17. Convergence sensitivity analysis
# ============================================================

sensitivity_rows = []


for rate in SENSITIVITY_CONVERGENCE_RATES:

    vehicle_value = (
        practical_convergence_vehicle(
            rate,
            k,
            REFERENCE_VEHICLE,
        )
    )


    penalty_value = (
        C
        + D
        * (
            1.0
            - rate
        )
    )


    sensitivity_rows.append(
        {
            "convergence_rate": rate,
            "vehicle_count": vehicle_value,
            "vehicle_count_ceiling": int(
                np.ceil(
                    vehicle_value
                )
            ),
            "penalty": penalty_value,
        }
    )


sensitivity_df = pd.DataFrame(
    sensitivity_rows
)


# ============================================================
# 18. Knee point 계산
#
# 중요:
# 실제 관측된 차량 범위 안에서만 계산한다.
# ============================================================

x_knee_curve = np.linspace(

    float(
        np.min(
            x_fit
        )
    ),

    float(
        np.max(
            x_fit
        )
    ),

    KNEE_GRID_SIZE,
)


y_knee_curve = fit_model(
    x_knee_curve,
    *params,
)


(
    knee_vehicle,
    knee_penalty,
    knee_distance,
) = geometric_knee_point(
    x_knee_curve,
    y_knee_curve,
)


knee_vehicle_integer = int(
    np.ceil(
        knee_vehicle
    )
)


knee_penalty_integer = float(
    fit_model(
        knee_vehicle_integer,
        *params,
    )
)


# Knee 부근에서 차량 한 대 추가 효과
knee_marginal_reduction = (
    one_vehicle_penalty_reduction(
        knee_vehicle_integer,
        D,
        k,
        REFERENCE_VEHICLE,
    )
)


# ============================================================
# 19. 터미널 결과 출력
# ============================================================

print(
    "=" * 75
)

print(
    "Penalty - Vehicle 지수 비선형 회귀 결과"
)

print(
    "=" * 75
)


print(
    "\n[Regression model]"
)

print(
    "P(V) = C + D * exp[-k(V - V0)]"
)

print(
    f"V0 = {REFERENCE_VEHICLE:.0f} vehicles"
)

print(
    f"C  = {C:,.0f}"
)

print(
    f"D  = {D:,.0f}"
)

print(
    f"k  = {k:.8f}"
)

print(
    f"Equivalent A = {A_equivalent:,.0f}"
)


# ------------------------------------------------------------

print(
    "\n[SciPy optimizer convergence]"
)

print(
    f"Converged = {optimizer_converged}"
)

print(
    "ftol = xtol = gtol = "
    f"{OPTIMIZER_TOLERANCE:.0e}"
)

print(
    f"Termination message = {message.strip()}"
)


# ------------------------------------------------------------

print(
    "\n[Goodness of fit - data used for fitting]"
)

print(
    f"R²    = {r_squared:.6f}"
)

print(
    f"RMSE  = {rmse:,.0f}"
)

print(
    f"NRMSE = {nrmse:.4%}"
)


# ------------------------------------------------------------

if len(
    x_excluded
) > 0:

    print(
        "\n[Reference - including excluded data]"
    )

    print(
        f"R²    = {r_squared_all:.6f}"
    )

    print(
        f"RMSE  = {rmse_all:,.0f}"
    )

    print(
        f"NRMSE = {nrmse_all:.4%}"
    )


# ------------------------------------------------------------

print(
    "\n[Approximate parameter confidence intervals]"
)


parameter_names = (
    "C",
    "D",
    "k",
)


for (
    name,
    value,
    standard_error,
    interval,
) in zip(
    parameter_names,
    params,
    parameter_standard_errors,
    parameter_ci,
):

    if name == "k":

        print(
            f"{name}: "
            f"{value:.8f} "
            f"(SE={standard_error:.8f}, "
            f"{CONFIDENCE_LEVEL:.0%} CI="
            f"{interval[0]:.8f} ~ "
            f"{interval[1]:.8f})"
        )

    else:

        print(
            f"{name}: "
            f"{value:,.0f} "
            f"(SE={standard_error:,.0f}, "
            f"{CONFIDENCE_LEVEL:.0%} CI="
            f"{interval[0]:,.0f} ~ "
            f"{interval[1]:,.0f})"
        )


# ------------------------------------------------------------

print(
    "\n[Practical convergence]"
)

print(
    "Definition:"
)

print(
    f"Penalty reduction potential at V0="
    f"{REFERENCE_VEHICLE:.0f} vehicles is "
    f"{PRIMARY_CONVERGENCE_RATE:.0%} exhausted."
)


print(
    f"\n{PRIMARY_CONVERGENCE_RATE:.0%} convergence vehicle "
    f"= {primary_convergence_vehicle:.2f}"
)

print(
    "Minimum integer vehicle count "
    f"= {primary_convergence_vehicle_integer}"
)


print(
    f"{CONFIDENCE_LEVEL:.0%} approximate CI "
    f"= {vehicle_ci[0]:.2f} "
    f"~ {vehicle_ci[1]:.2f} vehicles"
)


print(
    "Penalty at practical convergence "
    f"= {primary_convergence_penalty:,.0f}"
)


print(
    f"{CONFIDENCE_LEVEL:.0%} approximate Penalty CI "
    f"= {penalty_ci[0]:,.0f} "
    f"~ {penalty_ci[1]:,.0f}"
)


if (
    primary_convergence_vehicle
    > np.max(x_fit)
):

    print(
        "\nWARNING:"
    )

    print(
        "Practical convergence is outside "
        "the observed vehicle range."
    )

    print(
        "Therefore this value is based on "
        "model extrapolation."
    )


# ------------------------------------------------------------

print(
    "\n[Convergence sensitivity]"
)


print(
    sensitivity_df.to_string(
        index=False,
        formatters={
            "convergence_rate":
                lambda value:
                f"{value:.0%}",

            "vehicle_count":
                lambda value:
                f"{value:.2f}",

            "vehicle_count_ceiling":
                lambda value:
                f"{value:d}",

            "penalty":
                lambda value:
                f"{value:,.0f}",
        },
    )
)


# ------------------------------------------------------------

print(
    "\n[Geometric knee point]"
)

print(
    f"Knee vehicle = {knee_vehicle:.2f}"
)

print(
    f"Integer candidate = {knee_vehicle_integer}"
)

print(
    f"Knee penalty = {knee_penalty:,.0f}"
)

print(
    f"Penalty at integer knee "
    f"= {knee_penalty_integer:,.0f}"
)

print(
    f"Expected reduction from "
    f"{knee_vehicle_integer} -> "
    f"{knee_vehicle_integer + 1} vehicles "
    f"= {knee_marginal_reduction:,.0f}"
)

print(
    "Normalized chord distance "
    f"= {knee_distance:.6f}"
)


# ============================================================
# 20. 메인 그래프
# ============================================================

observed_x_min = float(
    np.min(
        x_fit
    )
)

observed_x_max = float(
    np.max(
        x_fit
    )
)


# Practical convergence가 데이터 범위를 조금 넘어가면
# 외삽구간까지 보여준다.
requested_plot_max = max(
    observed_x_max,
    primary_convergence_vehicle * 1.05,
)


# 단, 과도한 외삽으로 그래프가 눌리는 것을 방지
plot_x_max = min(
    requested_plot_max,
    observed_x_max
    * MAX_PLOT_EXTRAPOLATION_FACTOR,
)


x_curve = np.linspace(
    observed_x_min,
    plot_x_max,
    CURVE_GRID_SIZE,
)


y_curve = fit_model(
    x_curve,
    *params,
)


# fitted mean curve의 confidence band
y_standard_error = (
    prediction_standard_error(
        x_curve,
        params,
        covariance,
        REFERENCE_VEHICLE,
    )
)


y_margin = (
    critical_value
    * y_standard_error
)


plt.figure(
    figsize=(
        13,
        7.5,
    )
)


# ------------------------------------------------------------
# 회귀에 사용한 실제 데이터
# ------------------------------------------------------------

plt.scatter(
    x_fit,
    y_fit,
    s=35,
    label="Data used for fitting",
    zorder=3,
)


# ------------------------------------------------------------
# 제외 데이터
# ------------------------------------------------------------

if len(
    x_excluded
) > 0:

    plt.scatter(
        x_excluded,
        y_excluded,
        s=60,
        marker="x",
        label="Excluded data",
        zorder=4,
    )


# ------------------------------------------------------------
# 실제 데이터 범위 내 회귀곡선
# ------------------------------------------------------------

observed_curve_mask = (
    x_curve
    <= observed_x_max
)


plt.plot(
    x_curve[
        observed_curve_mask
    ],

    y_curve[
        observed_curve_mask
    ],

    linewidth=2.2,

    label="Exponential regression",
)


# ------------------------------------------------------------
# 외삽 부분은 점선
# ------------------------------------------------------------

extrapolation_mask = (
    x_curve
    > observed_x_max
)


if np.any(
    extrapolation_mask
):

    x_extra = np.concatenate(
        (
            [
                observed_x_max
            ],

            x_curve[
                extrapolation_mask
            ],
        )
    )


    y_extra = fit_model(
        x_extra,
        *params,
    )


    plt.plot(
        x_extra,
        y_extra,
        linewidth=2.0,
        linestyle="--",
        label="Model extrapolation",
    )


# ------------------------------------------------------------
# 95% confidence band
# ------------------------------------------------------------

plt.fill_between(
    x_curve,

    y_curve
    - y_margin,

    y_curve
    + y_margin,

    alpha=0.18,

    label=(
        f"{CONFIDENCE_LEVEL:.0%} "
        "confidence band of fitted mean"
    ),
)


# ------------------------------------------------------------
# 점근선 C
# ------------------------------------------------------------

plt.axhline(
    C,
    linestyle=":",
    linewidth=1.6,
    label=(
        f"Asymptote C = "
        f"{C:,.0f}"
    ),
)


# ------------------------------------------------------------
# Knee point
# ------------------------------------------------------------

plt.axvline(
    knee_vehicle,
    linestyle="-.",
    linewidth=1.7,
    label=(
        f"Geometric knee = "
        f"{knee_vehicle:.2f}"
    ),
)


plt.scatter(
    [
        knee_vehicle
    ],
    [
        knee_penalty
    ],
    s=70,
    zorder=5,
)


# ------------------------------------------------------------
# Practical convergence
# ------------------------------------------------------------

if (
    primary_convergence_vehicle
    <= plot_x_max
):

    plt.axvline(
        primary_convergence_vehicle,
        linestyle="--",
        linewidth=1.7,
        label=(
            f"{PRIMARY_CONVERGENCE_RATE:.0%} "
            "practical convergence = "
            f"{primary_convergence_vehicle:.2f}"
        ),
    )


    plt.scatter(
        [
            primary_convergence_vehicle
        ],
        [
            primary_convergence_penalty
        ],
        s=70,
        zorder=5,
    )


else:

    print(
        "\nGraph warning:"
    )

    print(
        f"{PRIMARY_CONVERGENCE_RATE:.0%} "
        "practical convergence "
        f"({primary_convergence_vehicle:.2f}) "
        "is outside the displayed range."
    )


# ------------------------------------------------------------
# 그래프 형식
# ------------------------------------------------------------

plt.xlabel(
    "Number of Vehicles"
)

plt.ylabel(
    "Penalty"
)

plt.title(
    "Penalty vs. Number of Vehicles: "
    "Exponential Regression"
)


plt.gca().yaxis.set_major_formatter(
    FuncFormatter(
        lambda value, _:
        f"{value:,.0f}"
    )
)


plt.grid(
    alpha=0.35
)

plt.legend()

plt.tight_layout()

# 메인 회귀 그래프 PNG 저장
plt.savefig(
    SAVE_PATH / "Penalty_vs_Vehicles.png",
    dpi=300,
    bbox_inches="tight",
)



# ============================================================
# 21. Residual diagnostic
# ============================================================

if SHOW_RESIDUAL_PLOT:

    order = np.argsort(
        x_fit
    )


    x_residual = x_fit[
        order
    ]


    residual_sorted = residuals[
        order
    ]


    plt.figure(
        figsize=(
            12,
            5.5,
        )
    )


    plt.scatter(
        x_residual,
        residual_sorted,
        s=35,
    )


    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1.5,
    )


    plt.xlabel(
        "Number of Vehicles"
    )

    plt.ylabel(
        "Residual = Actual - Fitted"
    )

    plt.title(
        "Residual Diagnostic"
    )


    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _:
            f"{value:,.0f}"
        )
    )


    plt.grid(
        alpha=0.35
    )

    plt.tight_layout()


# ============================================================
# 22. 지수함수 선형화 diagnostic
# ============================================================

if SHOW_LINEARIZED_DIAGNOSTIC:

    # --------------------------------------------------------
    # 원래 모델:
    #
    # P(V) - C
    # = D exp[-k(V-V0)]
    #
    #
    # 양변을 D로 나누고 ln을 취하면
    #
    # ln((P-C)/D)
    # = -k(V-V0)
    #
    # 즉 직선 형태가 된다.
    # --------------------------------------------------------

    valid_log_mask = (
        y_fit > C
    )


    if np.count_nonzero(
        valid_log_mask
    ) >= 2:

        x_linearized = (
            x_fit[
                valid_log_mask
            ]
            - REFERENCE_VEHICLE
        )


        y_linearized = np.log(
            (
                y_fit[
                    valid_log_mask
                ]
                - C
            )
            / D
        )


        x_linearized_line = np.linspace(
            0.0,
            float(
                np.max(
                    x_linearized
                )
            ),
            500,
        )


        y_linearized_line = (
            -k
            * x_linearized_line
        )


        plt.figure(
            figsize=(
                12,
                5.5,
            )
        )


        plt.scatter(
            x_linearized,
            y_linearized,
            s=35,
            label=(
                "Transformed fitting data"
            ),
        )


        plt.plot(
            x_linearized_line,
            y_linearized_line,
            linewidth=2.0,
            label=(
                "Ideal fitted line: "
                f"y = -{k:.6f}x"
            ),
        )


        plt.xlabel(
            "Vehicles above "
            f"V0 = {REFERENCE_VEHICLE:.0f}"
        )


        plt.ylabel(
            "ln((Penalty - C) / D)"
        )


        plt.title(
            "Linearized Exponential-Model Diagnostic"
        )


        plt.grid(
            alpha=0.35
        )

        plt.legend()

        plt.tight_layout()


    else:

        print(
            "\nLinearized diagnostic "
            "cannot be generated:"
        )

        print(
            "There are fewer than "
            "2 observations satisfying Penalty > C."
        )


# ============================================================
# 23. 모든 그래프 출력
# ============================================================

plt.show()