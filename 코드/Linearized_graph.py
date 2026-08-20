import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 데이터 불러오기
df = pd.read_csv("C:/Users/choih/Desktop/연구/PDPTW/자료/결과/Ortools/Penalty_per_vehicles/penalty_per_vehicles.csv")

x = df["num_vehicles"].to_numpy()
y = df["total_penalty"].to_numpy()


# -------------------------
# 이상 구간 제외
# 예: 147~164대 구간 제외
# -------------------------
mask = ~((x >= 147) & (x <= 164))

x_fit = x[mask]
y_fit = y[mask]


# -------------------------
# 비선형 모델
# P(V) = C + A * exp(-kV)
# -------------------------
def exponential_model(V, C, A, k):
    return C + A * np.exp(-k * V)


# 초기 추정값
p0 = [
    1.2e9,   # C
    8.0e9,   # A
    0.02     # k
]


params, covariance = curve_fit(
    exponential_model,
    x_fit,
    y_fit,
    p0=p0,
    maxfev=100000
)

C, A, k = params

print(f"C = {C:,.0f}")
print(f"A = {A:,.0f}")
print(f"k = {k:.6f}")


# -------------------------
# 회귀 곡선
# -------------------------
x_curve = np.linspace(x.min(), x.max(), 500)
y_curve = exponential_model(x_curve, C, A, k)


plt.figure(figsize=(12, 7))

plt.scatter(x, y, label="Original data")
plt.plot(
    x_curve,
    y_curve,
    linewidth=2,
    label="Nonlinear regression"
)

plt.xlabel("Number of Vehicles")
plt.ylabel("Penalty")
plt.legend()
plt.grid()

plt.show()