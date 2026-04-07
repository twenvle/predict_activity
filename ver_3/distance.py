import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# 1. データの準備 (標準化済みのものを使ってください)
# 例: 5つの候補化合物, 4つの記述子
X = np.array(
    [
        [1.2, 0.5, 3.1, 0.1],  # 化合物A
        [1.1, 0.6, 3.0, 0.2],  # 化合物B (Aに激似)
        [3.5, 2.2, 0.1, 0.9],  # 化合物C (全然違う)
        [0.1, 0.1, 0.1, 0.1],  # 化合物D
        [1.2, 0.5, 3.1, 0.15],  # 化合物E (Aにかなり似ている)
    ]
)
names = ["A", "B", "C", "D", "E"]

# --- 方法1: ユークリッド距離行列の計算 ---
# 全組み合わせの距離を一発で計算します
dist_matrix = euclidean_distances(X, X)

# 見やすくDataFrameにする
df_dist = pd.DataFrame(dist_matrix, index=names, columns=names)

print("【距離行列】(0に近いほど似ている)")
print(df_dist.round(2))

# --- 方法2: カーネル類似度の計算 (ガウス過程の視点) ---
# モデルが読み込まれている前提 (model)
# RBFカーネルの値: exp(-gamma * distance^2)
# ※学習済みモデルからカーネル関数を取り出して計算させます
model = joblib.load(
    r"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_3\gauss_ver1.pkl"
)
kernel_matrix = model.kernel_(X)

df_sim = pd.DataFrame(kernel_matrix, index=names, columns=names)

print("\n【カーネル類似度】(1に近いほど似ている)")
print(df_sim.round(2))

# --- 可視化 (ヒートマップ) ---
# 距離行列を色で見るのが一番わかりやすいです
plt.figure(figsize=(6, 5))
sns.heatmap(df_dist, annot=True, cmap="viridis", fmt=".2f")
plt.title("Distance Matrix (Blue is similar)")
plt.show()
