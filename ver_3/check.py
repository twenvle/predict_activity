import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import matplotlib.pyplot as plt

# 1. データ読み込み（ここは同じ）
df = pd.read_csv(
    r"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_3\out\save\value.csv"
)
df = df.dropna(how="any").reset_index(drop=True)

descripters = [
    "delta_g_hartree",
    "dipole_moment_debye",
    "gap_ev",
    "molecular_volume_A3",
    "nbo_charge",
]

# 2. 全ての記述子を一度に使って学習させる
X = df[descripters]
y = df["yield"]

# スケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# カーネル定義 (ARD: Anisotropic RBF)
# 特徴量の数だけ length_scale を用意する
length = len(descripters)
kernel = (
    ConstantKernel(1.0) * RBF([1.0] * length, length_scale_bounds=(1e-2, 1e10))
    + WhiteKernel()
)

model = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=20, random_state=42
)

# 全データでfit
model.fit(X_scaled, y)

# 3. 学習されたカーネルパラメータ（Length Scale）を取り出す
# model.kernel_ は複合カーネルなので、RBFの部分を取り出す必要があります
# 構造: Constant * RBF + White -> k1 (Constant*RBF) + k2 (White) -> k1.k2 (RBF)
learned_kernel = model.kernel_
rbf_kernel_params = learned_kernel.k1.k2.length_scale

print("-" * 50)
print(f"{'Descriptor':<25} | {'Length Scale':<15} | {'Importance'}")
print("-" * 50)

for name, scale in zip(descripters, rbf_kernel_params):
    # scaleが小さいほど重要。大きい(1e5以上など)と無視されている。
    importance = "効いているかも" if scale < 100 else "無視されている"
    print(f"{name:<25} | {scale:.2e}        | {importance}")

print("-" * 50)
print("Log-Marginal-Likelihood:", model.log_marginal_likelihood())
