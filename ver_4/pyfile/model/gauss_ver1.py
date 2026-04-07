import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib


def make_model(name):
    df = pd.read_csv(
        rf"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\csvfile\known\{name}.csv"
    )

    df = df.dropna(how="any").reset_index(drop=True)

    descriptors = [
        "dipole_moment_debye",
        "3and6",
        "lumo_ev",
        "homo_ev",
    ]

    length = len(descriptors)
    X = df[descriptors]
    y = df["yield"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kernel = (
        ConstantKernel() * RBF([1.0] * length, length_scale_bounds=(1e-2, 1e10))
        + WhiteKernel()
    )
    model = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=10, random_state=42
    )
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_index, test_index in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)

        y_pred.append(model.predict(X_test)[0])
        y_true.append(y_test.values[0])

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    model.fit(X_scaled, y)

    content = {"model": model, "scaler": scaler}

    print(r2)
    print(mae)

    joblib.dump(
        content,
        rf"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\pklfile\{name}.pkl",
    )
