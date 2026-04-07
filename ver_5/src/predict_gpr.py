import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors


def calculate_ei(
    mu: np.ndarray, sigma: np.ndarray, current_best: float, xi: float
) -> np.ndarray:
    sigma_safe = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    improvement = np.asarray(mu, dtype=float) - float(current_best) - float(xi)
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(sigma_safe <= 1e-12, 0.0, ei)
    return ei


def calculate_pi(
    mu: np.ndarray, sigma: np.ndarray, current_best: float, xi: float
) -> np.ndarray:
    sigma_safe = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    z = (np.asarray(mu, dtype=float) - float(current_best) - float(xi)) / sigma_safe
    pi = norm.cdf(z)
    pi = np.where(sigma_safe <= 1e-12, 0.0, pi)
    return pi


def calculate_ucb(mu: np.ndarray, sigma: np.ndarray, beta: float) -> np.ndarray:
    return np.asarray(mu, dtype=float) + float(beta) * np.asarray(sigma, dtype=float)


def compute_train_knn_distances(X_train_scaled: np.ndarray, k: int) -> np.ndarray:
    n_samples = X_train_scaled.shape[0]
    effective_k = min(max(1, k), n_samples - 1)  # 自分自身を除くため n_samples - 1
    nn = NearestNeighbors(
        n_neighbors=effective_k + 1, metric="euclidean"
    )  # 自分自身を含めて k+1 に設定, metricは距離の種類
    nn.fit(X_train_scaled)  # 計算を速くするための処理

    # 全trainの中から、各点ごとに最も近い K+1 点を選んでいる
    # distances: 距離の値
    # 各行 -> 1つのクエリ点
    # 各列 -> 近い順の距離
    # distances = [[0.0, d_1, d_2, ..., d_k],...,  最初は自分自身の距離、その後に近い順の距離が続く
    #              [0.0, d_1, d_2, ..., d_k]]
    # indices: 近い順のインデックス
    distances, _ = nn.kneighbors(X_train_scaled, return_distance=True)
    # 0 番目は自分自身なので除く
    return distances[:, 1 : effective_k + 1].mean(axis=1)


def compute_query_knn_distances(
    X_train_scaled: np.ndarray, X_query_scaled: np.ndarray, k: int
) -> np.ndarray:
    n_samples = X_train_scaled.shape[0]
    effective_k = min(max(1, k), n_samples)
    nn = NearestNeighbors(n_neighbors=effective_k, metric="euclidean")
    nn.fit(X_train_scaled)
    distances, _ = nn.kneighbors(X_query_scaled, return_distance=True)
    return distances.mean(axis=1)


def compute_ad_penalty(ad_ratio: np.ndarray, mode: str) -> np.ndarray:
    ad_ratio = np.asarray(ad_ratio, dtype=float)
    if mode == "none":
        return np.ones_like(ad_ratio)
    if mode == "hard":
        return (ad_ratio <= 1.0).astype(float)
    # soft
    return np.exp(-(ad_ratio**2))


def transform_features(X: pd.DataFrame, scaler: Any | None) -> np.ndarray:
    if scaler is None:
        return X.to_numpy(dtype=float)
    return scaler.transform(X)


def attach_known_flags(
    candidate_df: pd.DataFrame, train_df: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    known_by_cas = pd.Series(False, index=candidate_df.index)
    known_by_smiles = pd.Series(False, index=candidate_df.index)

    if "cas" in candidate_df.columns and "cas" in train_df.columns:
        train_cas = set(train_df["cas"].dropna().astype(str))
        known_by_cas = candidate_df["cas"].astype(str).isin(train_cas)

    if "smiles" in candidate_df.columns and "smiles" in train_df.columns:
        train_smiles = set(train_df["smiles"].dropna().astype(str))
        known_by_smiles = candidate_df["smiles"].astype(str).isin(train_smiles)

    return known_by_cas, known_by_smiles


def predict(
    train_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    summary_path: Path,
    joblib_path: Path,
    method: str,
    target: str = "yield",
    xi: float = 0.5,
    beta: float = 1.0,
    ci_level: float = 0.95,
    ad_k: int = 5,
    ad_quantile: float = 0.95,
    ad_mode: str = "soft",
    sort_by: str = "ei_ad",
):
    today = date.today().strftime("%Y%m%d")
    estimator = joblib.load(joblib_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    sigma_scale = summary["final_best_on_full_data"].get("sigma_", 1.0)
    # named_steps is only available if the estimator is a Pipeline.
    model = estimator.named_steps[method]
    scaler = estimator.named_steps["scaler"]
    descriptors = summary["descriptors"]

    train_df = train_df.dropna(how="any").reset_index(drop=True)
    current_best_y = train_df[target].max()

    output_df = candidate_df.copy()
    output_df["prediction_status"] = "missing_descriptor"

    valid_mask = candidate_df[descriptors].notna().all(axis=1)
    valid_idx = output_df.index[valid_mask]

    known_by_cas, known_by_smiles = attach_known_flags(candidate_df, train_df)
    known_any = known_by_cas | known_by_smiles
    output_df["known_any"] = known_any

    if len(valid_idx) > 0:
        X_train = train_df[descriptors].astype(float)
        X_train_scaled = transform_features(X_train, scaler)

        X_query = candidate_df.loc[valid_mask, descriptors].astype(float)
        X_query_scaled = transform_features(X_query, scaler)

        mu, sigma_raw = model.predict(X_query_scaled, return_std=True)
        sigma_raw = np.asarray(sigma_raw, dtype=float)
        sigma_cal = sigma_raw * float(sigma_scale)

        # 適用領域 (AD): scaled 空間での kNN 距離
        train_knn_dist = compute_train_knn_distances(X_train_scaled, ad_k)
        ad_threshold = float(
            np.quantile(train_knn_dist, ad_quantile)
        )  # np.quantile は配列の指定した分位数を計算する関数
        query_knn_dist = compute_query_knn_distances(
            X_train_scaled, X_query_scaled, ad_k
        )
        ad_ratio = query_knn_dist / max(ad_threshold, 1e-12)
        ad_in = ad_ratio <= 1.0
        ad_penalty = compute_ad_penalty(ad_ratio, ad_mode)

        ei = calculate_ei(mu=mu, sigma=sigma_cal, current_best=current_best_y, xi=xi)
        pi = calculate_pi(mu=mu, sigma=sigma_cal, current_best=current_best_y, xi=xi)
        ucb = calculate_ucb(mu=mu, sigma=sigma_cal, beta=beta)
        ei_ad = ei * ad_penalty

        valid_known = known_any.loc[valid_idx].to_numpy(dtype=bool)
        ei = np.where(valid_known, 0.0, ei)
        pi = np.where(valid_known, 0.0, pi)
        ei_ad = np.where(valid_known, 0.0, ei_ad)

        alpha = 1.0 - float(ci_level)
        z = float(norm.ppf(1.0 - alpha / 2.0))
        ci_lower = mu - z * sigma_cal
        ci_upper = mu + z * sigma_cal

        result = pd.DataFrame(
            {
                "prediction_status": "ok",
                "mu": mu,
                "sigma_raw": sigma_raw,
                "sigma_cal": sigma_cal,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "current_best_y": current_best_y,
                "xi": float(xi),
                "beta": float(beta),
                "pi": pi,
                "ei": ei,
                "ucb": ucb,
                "ad_knn_distance": query_knn_dist,
                "ad_threshold": ad_threshold,
                "ad_ratio": ad_ratio,
                "in_ad": ad_in,
                "ad_penalty": ad_penalty,
                "ei_ad": ei_ad,
            },
            index=valid_idx,
        )

        output_df.loc[valid_idx, result.columns] = result

    if sort_by not in output_df.columns:
        raise ValueError(f"sort_by='{sort_by}' 列が見つかりません。")

    output_df = output_df.sort_values(
        by=[sort_by, "mu" if "mu" in output_df.columns else sort_by],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    output_path = Path(__file__).resolve().parent / f"out/{today}_{method}_output.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
