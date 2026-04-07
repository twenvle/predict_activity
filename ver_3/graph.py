import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

# 1. データ読み込み
df = pd.read_csv(
    r"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_3\out\save\value.csv"
)
df = df.dropna(how="any").reset_index(drop=True)

# ★選抜された3つの記述子のみを使用
selected_descriptors = ["nbo_charge", "dipole_moment_debye", "delta_g_hartree"]

X = df[selected_descriptors]
y = df["yield"]

# --- 部分1: 散布図で可視化（自分の目で相関を確認する） ---
plt.figure(figsize=(15, 5))

for i, col in enumerate(selected_descriptors):
    plt.subplot(1, 3, i + 1)
    plt.scatter(X[col], y, alpha=0.7, color="blue")
    plt.title(f"{col} vs Yield")
    plt.xlabel(col)
    plt.ylabel("Yield")
    plt.grid(True)

plt.tight_layout()
plt.show()  # ここでグラフが表示されます。形をチェックしてください！

# --- 部分2: 選抜メンバーだけでGPR予測 ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

length = len(selected_descriptors)
kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * RBF([1.0] * length, length_scale_bounds=(1e-1, 1e5))  # 範囲を少し常識的に修正
    + WhiteKernel()
)

model = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=10, random_state=42
)

loo = LeaveOneOut()
y_true, y_pred = [], []

# ここでも警告は無視してOK
import warnings
from sklearn.exceptions import ConvergenceWarning

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for train_index, test_index in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred.append(model.predict(X_test)[0])
        y_true.append(y_test.values[0])

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"選抜された記述子: {selected_descriptors}")
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")
