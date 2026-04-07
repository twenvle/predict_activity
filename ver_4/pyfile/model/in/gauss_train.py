import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.model_selection import LeaveOneOut, GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
import itertools
import warnings
from sklearn.exceptions import ConvergenceWarning

# ConvergenceWarning を無視する設定
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def predict(name):
    df = pd.read_csv(
        r"C:\Users\nabae\Desktop\kikuchi\pka_activity\ver4\3rd_20260223.csv"
    )

    df = df.dropna(how="any").reset_index(drop=True)

    descripters = [
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

    with open(
        rf"C:\Users\nabae\Desktop\kikuchi\pka_activity\ver4\out\descriptors\{name}.csv",
        "w",
        newline="\n",
    ) as f:
        f.write("descriptor, kernel, rmse, mae, r2\n")
        for i in range(1, 5):
            for combi in itertools.combinations(descripters, i):
                combi = list(combi)
                length = len(combi)
                X = df[combi]
                y = df["yield"]

                pipeline = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "gpr",
                            GaussianProcessRegressor(
                                n_restarts_optimizer=10, random_state=42
                            ),
                        ),
                    ]
                )
                kernels = [
                    # パターン1: RBFカーネル (非常に滑らかな関数を想定)
                    ConstantKernel(1.0, (1e-3, 1e3))
                    * RBF([1.0] * length, length_scale_bounds=(1e-2, 1e2))
                    + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),
                    # パターン2: Maternカーネル nu=2.5 (RBFより少しだけ粗い変化を許容)
                    ConstantKernel(1.0, (1e-3, 1e3))
                    * Matern([1.0] * length, length_scale_bounds=(1e-2, 1e2), nu=2.5)
                    + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),
                    # パターン3: Maternカーネル nu=1.5 (さらに粗い変化を許容)
                    ConstantKernel(1.0, (1e-3, 1e3))
                    * Matern([1.0] * length, length_scale_bounds=(1e-2, 1e2), nu=1.5)
                    + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),
                ]
                param_grid = {"gpr__kernel": kernels}
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

                final_clf = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    cv=inner_cv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                )
                final_clf.fit(X, y)

                best_kernel = final_clf.best_estimator_.named_steps["gpr"].kernel_

                combi_text = "|".join(combi)
                f.write(f'{combi_text},"{best_kernel}",{rmse},{mae},{r2}\n')
                print(
                    f"{combi_text},  kernel: {best_kernel},  rmse: {rmse},  mae: {mae},  r2: {r2}"
                )
