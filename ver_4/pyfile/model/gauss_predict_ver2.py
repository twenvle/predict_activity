import numpy as np
import pandas as pd
from scipy.stats import norm
import joblib


def make_csv(train_name, pkl_name, descriptors=None, xi=1):
    df = pd.read_csv(
        r"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\csvfile\unknown\unknown_data.csv"
    )

    df_train = pd.read_csv(
        rf"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\csvfile\known\{train_name}.csv"
    )

    content = joblib.load(
        rf"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\pklfile\{pkl_name}.pkl"
    )

    model = content["model"]
    scaler = content["scaler"]

    df = df.dropna(how="any").reset_index(drop=True)

    X = df[descriptors]
    y_train = df_train["yield"]

    X_scaled = scaler.transform(X)

    # 実測値の最大値
    current_best_y = np.max(y_train)

    # mu: 平均値, sigma: 標準偏差
    mu, sigma = model.predict(X_scaled, return_std=True)

    # EI = (mu - current_best - xi) * Phi(Z) + sigma * phi(Z)
    def calculate_ei(mu, sigma, current_best, xi=xi):
        with np.errstate(divide="warn"):
            imp = mu - current_best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0  # sigmaが0（すでに実験済みなど）ならEIは0
        return ei

    ei_scores = calculate_ei(mu, sigma, current_best_y)

    df["mu"] = mu
    df["sigma"] = sigma
    df["ei"] = ei_scores

    df.to_csv(
        rf"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\csvfile\predict\{pkl_name}.csv",
        index=False,
    )
