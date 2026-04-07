import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ==========================================
# 1. データの読み込み
# ==========================================
df = pd.read_csv(
    r"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_3\out\save\value.csv"
)

df = df.dropna(how="any").reset_index(drop=True)

X = df.drop(["cas", "smiles", "h1", "h2", "yield"], axis=1)  # 記述子（ダミー変数含む）
y = df["yield"]  # 目的変数（活性）

feature_names = X.columns  # 後で「どの記述子が選ばれたか」見るために名前を保存

# ==========================================
# 2. 前処理（標準化）
# ==========================================
# LASSOでは、スケールの大きな変数に係数が引っ張られないよう標準化が「必須」です
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. モデル定義（LASSO CV）
# ==========================================
# cv=LeaveOneOut() にすることで、内部で最適なα(ペナルティ強度)を
# 23サンプルのLOOCVに基づいて自動決定します。
model = LassoCV(cv=LeaveOneOut(), random_state=42, max_iter=10000)

# ==========================================
# 4. 評価（外側のLOOCV: 予測性能の確認）
# ==========================================
# ※ LassoCV内部のLOOCVは「αを決めるため」、
#    ここのLOOCVは「モデルの予測精度(RMSE)を測るため」に行います。
loo = LeaveOneOut()
y_true_list = []
y_pred_list = []

print("LOOCV評価を開始します...")
for train_index, test_index in loo.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 学習 (この中で最適なαが毎回選ばれます)
    model.fit(X_train, y_train)

    # 予測
    pred = model.predict(X_test)

    y_true_list.append(y_test.values[0])
    y_pred_list.append(pred[0])

# 精度算出
rmse = np.sqrt(mean_squared_error(y_true_list, y_pred_list))
r2 = r2_score(y_true_list, y_pred_list)

print("------------------------------------------------")
print(f"LASSO LOOCV 結果 (N={len(y)}):")
print(f"  RMSE (誤差の大きさ): {rmse:.4f}")
print(f"  R2   (決定係数)    : {r2:.4f}")
print("------------------------------------------------")

# ==========================================
# 5. 全データでの最終学習と「変数選択結果」の確認
# ==========================================
print("\n全データで最終モデルを作成中...")
model.fit(X_scaled, y)

# 係数(coef_)を取得
coefs = pd.Series(model.coef_, index=feature_names)

print("\n【重要】LASSOによる変数選択の結果:")
print("------------------------------------------------")
# 係数が0でない（＝モデルが重要と判断した）記述子だけを表示
selected_features = coefs[coefs != 0].sort_values(key=abs, ascending=False)

if len(selected_features) == 0:
    print("警告: すべての係数が0になりました（有効な記述子が見つかりませんでした）。")
else:
    for name, val in selected_features.items():
        print(f"  {name:20s}: {val:.4f}")

    print("\n  ※ 値がプラスなら活性向上、マイナスなら活性低下に寄与")
    print("  ※ 値の絶対値が大きいほど影響力が強い")
print("------------------------------------------------")
