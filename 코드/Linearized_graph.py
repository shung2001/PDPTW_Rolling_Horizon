# -*- coding: utf-8 -*-
"""차량 대수에 따른 penalty의 지수 회귀와 수렴 지점을 시각화한다."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = (
    BASE_DIR.parent
    / "자료"
    / "결과"
    / "Ortools"
    / "Penalty_per_vehicles"
    / "penalty_per_vehicles.csv"
)

TOLERANCE = 1e-8
CONFIDENCE_Z = 1.959963984540054  # 양측 95% 신뢰구간
CONVERGENCE_RATE = 0.95


def exponential_model(vehicle_count, C, A, k):
    """P(V) = C + A exp(-kV)."""
    return C + A * np.exp(-k * vehicle_count)


def prediction_standard_error(vehicle_count, params, covariance):
    """Delta method로 회귀 평균의 표준오차를 계산한다."""
    C, A, k = params
    vehicle_count = np.asarray(vehicle_count, dtype=float)
    exp_term = np.exp(-k * vehicle_count)
    jacobian = np.column_stack(
        (
            np.ones_like(vehicle_count),
            exp_term,
            -A * vehicle_count * exp_term,
        )
    )
    variance = np.einsum("ij,jk,ik->i", jacobian, covariance, jacobian)
    return np.sqrt(np.maximum(variance, 0.0))


df = pd.read_csv(CSV_PATH)
x = pd.to_numeric(df["num_vehicles"], errors="raise").to_numpy(dtype=float)
y = pd.to_numeric(df["total_penalty"], errors="raise").to_numpy(dtype=float)

# 이상 구간 제외: 147~164대
mask = ~((x >= 147) & (x <= 164))
x_fit = x[mask]
y_fit = y[mask]

p0 = [1.2e9, 8.0e9, 0.02]
params, covariance, info, message, ier = curve_fit(
    exponential_model,
    x_fit,
    y_fit,
    p0=p0,
    ftol=TOLERANCE,
    xtol=TOLERANCE,
    gtol=TOLERANCE,
    maxfev=100000,
    full_output=True,
)

C, A, k = params
optimizer_converged = ier in (1, 2, 3, 4)
if not optimizer_converged:
    raise RuntimeError(f"회귀 최적화가 수렴하지 않았습니다(ier={ier}): {message}")
if k <= 0:
    raise ValueError(f"감쇠계수 k가 양수가 아니어서 수렴 차량 대수를 계산할 수 없습니다: {k}")

# 초기 감소 가능량 A의 95%가 감소한 지점:
# P(V95) - C = (1 - 0.95) A, 따라서 V95 = -ln(0.05) / k.
remaining_rate = 1.0 - CONVERGENCE_RATE
convergence_vehicle = -np.log(remaining_rate) / k
convergence_vehicle_integer = int(np.ceil(convergence_vehicle))
convergence_penalty = float(exponential_model(convergence_vehicle, C, A, k))

# V95는 k의 함수이므로 delta method로 차량 대수의 95% 신뢰구간을 계산한다.
k_variance = max(float(covariance[2, 2]), 0.0)
vehicle_derivative_k = np.log(remaining_rate) / (k**2)
vehicle_standard_error = abs(vehicle_derivative_k) * np.sqrt(k_variance)
vehicle_ci = (
    convergence_vehicle - CONFIDENCE_Z * vehicle_standard_error,
    convergence_vehicle + CONFIDENCE_Z * vehicle_standard_error,
)
vehicle_ci_integer = (
    max(0, int(np.floor(vehicle_ci[0]))),
    max(0, int(np.ceil(vehicle_ci[1]))),
)

penalty_gradient = np.array([1.0, remaining_rate, 0.0])
penalty_variance = max(float(penalty_gradient @ covariance @ penalty_gradient), 0.0)
penalty_margin = CONFIDENCE_Z * np.sqrt(penalty_variance)
penalty_ci = (
    convergence_penalty - penalty_margin,
    convergence_penalty + penalty_margin,
)

print(f"C = {C:,.0f}")
print(f"A = {A:,.0f}")
print(f"k = {k:.6f}")
print(f"최적화 수렴 여부: {optimizer_converged} (ftol/xtol/gtol={TOLERANCE:.0e})")
print(f"수렴 종료 사유: {message.strip()}")
print(
    f"Penalty 감소분의 95% 수렴 차량 대수: {convergence_vehicle:.2f}대 "
    f"(운영에 필요한 최소 정수 차량: {convergence_vehicle_integer}대)"
)
print(
    f"95% 신뢰구간의 차량 대수: {vehicle_ci[0]:.2f}~{vehicle_ci[1]:.2f}대 "
    f"(정수 범위: {vehicle_ci_integer[0]}~{vehicle_ci_integer[1]}대)"
)
print(
    f"95% 수렴 지점의 penalty: {convergence_penalty:,.0f} "
    f"(95% 신뢰구간: {penalty_ci[0]:,.0f}~{penalty_ci[1]:,.0f})"
)

x_curve = np.linspace(x.min(), x.max(), 500)
y_curve = exponential_model(x_curve, C, A, k)
y_standard_error = prediction_standard_error(x_curve, params, covariance)
y_margin = CONFIDENCE_Z * y_standard_error

plt.figure(figsize=(12, 7))
plt.scatter(x, y, label="Original data")
plt.plot(x_curve, y_curve, linewidth=2, label="Exponential regression")
plt.fill_between(
    x_curve,
    y_curve - y_margin,
    y_curve + y_margin,
    alpha=0.2,
    label="95% confidence interval",
)
plt.axvline(
    convergence_vehicle,
    color="tab:red",
    linestyle="--",
    label=f"95% convergence: {convergence_vehicle:.2f} vehicles",
)
plt.scatter(
    [convergence_vehicle],
    [convergence_penalty],
    color="tab:red",
    zorder=3,
)
plt.xlabel("Number of Vehicles")
plt.ylabel("Penalty")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
