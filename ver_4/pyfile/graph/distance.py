import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


df = pd.read_csv(
    r"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\csvfile\unknown_1-200_ver1.csv"
)

content = joblib.load(
    r"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\pklfile\gauss_ver1.pkl"
)

model = content["model"]
scaler = content["scaler"]

df = df.dropna(how="any").reset_index(drop=True)
df = df.sort_values(by="sigma", ascending=False)
df = df.head(20)

descriptors = [
    "dipole_moment_debye",
    "3and6",
    "lumo_ev",
    "homo_ev",
]

X = df[descriptors]
X_scaled = scaler.transform(X)

X_np = np.array(X_scaled)
names = df["cas"]

# --- 方法1: ユークリッド距離行列の計算 ---
# 全組み合わせの距離を一発で計算します
dist_matrix = euclidean_distances(X_np, X_np)

# 見やすくDataFrameにする
df_dist = pd.DataFrame(dist_matrix, index=names, columns=names)

# print("【距離行列】(0に近いほど似ている)")
# print(df_dist.round(2))

# --- 方法2: カーネル類似度の計算 (ガウス過程の視点) ---
# モデルが読み込まれている前提 (model)
# RBFカーネルの値: exp(-gamma * distance^2)
# ※学習済みモデルからカーネル関数を取り出して計算させます
kernel_matrix = model.kernel_(X_np)

df_sim = pd.DataFrame(kernel_matrix, index=names, columns=names)

# print("\n【カーネル類似度】(1に近いほど似ている)")
# print(df_sim.round(2))

variance = (
    kernel_matrix.diagonal().mean()
)  # 全て同じ値のはずですが念のため平均をとります

# 全体を分散で割って正規化する
df_sim_normalized = df_sim / variance


# --- 可視化 (ヒートマップ) ---
# 距離行列を色で見るのが一番わかりやすいです
plt.figure(figsize=(6, 5))
sns.heatmap(df_sim_normalized, annot=True, cmap="viridis", fmt=".2f")
plt.title("Distance Matrix (Blue is similar)")
plt.show()

# 対角成分（自分自身の分散）を取得
