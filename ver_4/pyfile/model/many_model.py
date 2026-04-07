import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, GridSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning

# 各種アルゴリズムのインポート
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def make_regularized_model(name, algorithm="ridge", rename=None, descriptors=None):
    if not rename:
        rename = f"{name}_{algorithm}"

    df = pd.read_csv(
        rf"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\csvfile\known\{name}.csv"
    )
    df = df.dropna(how="any").reset_index(drop=True)

    X = df[descriptors]
    y = df["yield"]

    # 手法に応じたモデルとハイパーパラメータの設定
    if algorithm == "ridge":
        estimator = Ridge(random_state=42)
        param_grid = {"model__alpha": np.logspace(-3, 3, 10)}

    elif algorithm == "lasso":
        estimator = Lasso(random_state=42, max_iter=10000)
        param_grid = {"model__alpha": np.logspace(-4, 2, 10)}

    elif algorithm == "bayesian":
        estimator = BayesianRidge()
        # ベイズ線形はパラメータ自動最適化が強力なため、グリッドサーチ範囲は控えめ
        param_grid = {"model__alpha_1": [1e-6, 1e-5], "model__lambda_1": [1e-6, 1e-5]}

    elif algorithm == "pls":
        estimator = PLSRegression()
        # 成分数は最大でも記述子数。過学習を防ぐため最大でも5程度を探る
        max_comp = min(6, len(descriptors) + 1)
        param_grid = {"model__n_components": list(range(1, max_comp))}

    else:
        raise ValueError("Unsupported algorithm")

    pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])

    outer_cv = LeaveOneOut()
    inner_cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    y_true, y_pred = [], []
    for train_index, test_index in outer_cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf.fit(X_train, y_train)
        y_pred.append(clf.predict(X_test)[0])
        y_true.append(y_test.values[0])

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {algorithm.upper()} Results ---")
    print(f"R2: {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    # 全データを用いた最終モデルの学習
    final_clf = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    final_clf.fit(X, y)

    print(f"Best Parameters: {final_clf.best_params_}")

    # 各アルゴリズム固有の解釈可能な情報の出力
    best_model = final_clf.best_estimator_.named_steps["model"]
    if algorithm in ["ridge", "lasso"]:
        print("\n[Feature Coefficients]")
        for desc, coef in zip(descriptors, best_model.coef_):
            # Lassoで係数が0になった記述子をわかりやすく表示
            if coef == 0.0:
                print(f"{desc}: Dropped (0.0)")
            else:
                print(f"{desc}: {coef:.4f}")

    # 保存処理
    scaler = final_clf.best_estimator_.named_steps["scaler"]
    content = {"model": best_model, "scaler": scaler}

    joblib.dump(
        content,
        rf"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\pklfile\{rename}.pkl",
    )


# 実行例
# descriptors_list = ["homo_ev", "lumo_ev", ..., "hba"]
# make_regularized_model("my_dataset", algorithm="lasso", descriptors=descriptors_list)
