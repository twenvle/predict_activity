import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import itertools
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def predict_combinations(name, algorithm="ols", max_comb=4):
    """
    algorithm: "ols" (線形回帰) または "svr_linear" (線形SVR)
    max_comb: 総当たりする記述子の最大数
    """
    # ※パスはご自身の環境に合わせて修正してください
    df = pd.read_csv(
        rf"C:\Users\nabae\Desktop\kikuchi\pka_activity\ver4\3rd_20260223.csv"
    )
    df = df.dropna(how="any").reset_index(drop=True)

    descriptors = [
        "homo_ev",
        "lumo_ev",
        "gap_ev",
        "omega",
        "dipole_moment_debye",
        "molecular_volume_A3",
        "h_nbo_charge",
        "o_nbo_charge",
        "polar",
        "sasa",
        "3and6",
        "4and5",
        "logp",
        "hbd",
        "hba",
    ]

    out_file = rf"C:\Users\nabae\Desktop\kikuchi\pka_activity\ver4\out\descriptors\{name}_{algorithm}.csv"

    # アルゴリズムの設定
    if algorithm == "ols":
        estimator = LinearRegression()
        param_grid = {}  # OLSはハイパーパラメータなし
    elif algorithm == "svr_linear":
        estimator = SVR(kernel="linear")
        param_grid = {"model__C": [0.1, 1, 10], "model__epsilon": [0.01, 0.1]}
    else:
        raise ValueError("Unsupported algorithm")

    with open(out_file, "w", newline="\n") as f:
        f.write("descriptor, best_params, rmse, mae, r2\n")

        # 1個からmax_comb個までの組み合わせを総当たり
        for i in range(1, max_comb + 1):
            for combi in itertools.combinations(descriptors, i):
                combi = list(combi)
                X = df[combi]
                y = df["yield"]

                pipeline = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", estimator),
                    ]
                )

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

                # 全データでの最終学習とベストパラメータの取得
                final_clf = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    cv=inner_cv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                )
                final_clf.fit(X, y)

                best_params = final_clf.best_params_ if param_grid else "None"
                combi_text = "|".join(combi)

                f.write(f'{combi_text},"{best_params}",{rmse},{mae},{r2}\n')
                print(
                    f"{combi_text}, params: {best_params}, rmse: {rmse:.3f}, mae: {mae:.3f}, r2: {r2:.3f}"
                )


# 実行例
# predict_combinations("my_dataset", algorithm="ols", max_comb=4)
