import numpy as np
import pandas as pd
from scipy.stats import norm
import joblib


df = pd.read_csv(
    r"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_3\out\save\unknown_1-200.csv"
)
df_train = pd.read_csv(
    r"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_3\out\save\value.csv"
)

# 1. モデルの読み込み (すでに学習済みのもの)
model = joblib.load(
    r"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_3\gauss_ver1.pkl"
)

df = df.dropna(how="any").reset_index(drop=True)

descriptors = [
    "dipole_moment_debye",
    "3and6",
    "lumo_ev",
    "homo_ev",
]

X = df[descriptors]
y_train = df_train["yield"]

# 現在の学習データの中での「最大の実測値（ベストスコア）」を取得しておく必要があります
# (EIの計算に「現状のベストをどれだけ超えそうか」という情報が必要なため)
current_best_y = np.max(y_train)

# 2. 未実験の候補データ (X_candidate) を用意
# ※本来はCSVなどから読み込みますが、ここでは仮のデータを作成します
# 例: 5つの未実験化合物、記述子は4つ

# 3. ガウス過程回帰で予測 (平均と標準偏差の両方を出す)
# return_std=True が必須です
mu, sigma = model.predict(X, return_std=True)

# 4. 獲得関数 (Expected Improvement) の計算
# 計算式: EI = (mu - current_best - xi) * Phi(Z) + sigma * phi(Z)
# 直感的な意味: 「予測値が高い」かつ「不確実性が大きい（大化けするかも）」ものを高く評価する


def calculate_ei(mu, sigma, current_best, xi=0.01):
    with np.errstate(divide="warn"):
        imp = mu - current_best - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0  # sigmaが0（すでに実験済みなど）ならEIは0
    return ei


ei_scores = calculate_ei(mu, sigma, current_best_y)

# 5. 結果の確認と提案
print("各候補の予測値とEIスコア:")
df["mu"] = None
df["sigma"] = None
df["ei"] = None
for i in range(len(X)):
    df.loc[i, "mu"] = mu[i]
    df.loc[i, "sigma"] = sigma[i]
    df.loc[i, "ei"] = ei_scores[i]

df.to_csv(
    r"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_3\out\predict\samples_1-100_ver1.csv",
    index=False,
)

"""
# 最もEIが高い(有望な)候補のインデックスを取得
best_candidate_index = np.argmax(ei_scores)
print(
    f"\n★ 次に実験すべき化合物は: 候補{df.loc[best_candidate_index, 'smiles']}:{mu[i]}"
)
"""
