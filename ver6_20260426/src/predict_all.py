import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors

_SIGMA_FLOOR = 1e-12
_VALID_METHODS = {"gpr", "ols", "ridge", "lasso", "elasticnet", "pls", "svr", "blr"}
TARGET = {"yield", "conversion", "selectivity"}
AdMode = Literal["none", "hard", "soft"]


# ── 獲得関数 ──────────────────────────────────────────────────────────────────


def calculate_ei(mu, sigma, current_best, xi):
    sigma_safe = np.maximum(np.asarray(sigma, dtype=float), _SIGMA_FLOOR)
    improvement = np.asarray(mu, dtype=float) - float(current_best) - float(xi)
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(sigma_safe <= _SIGMA_FLOOR, 0.0, ei)


def calculate_pi(mu, sigma, current_best, xi):
    sigma_safe = np.maximum(np.asarray(sigma, dtype=float), _SIGMA_FLOOR)
    z = (np.asarray(mu, dtype=float) - float(current_best) - float(xi)) / sigma_safe
    pi = norm.cdf(z)
    return np.where(sigma_safe <= _SIGMA_FLOOR, 0.0, pi)


def calculate_ucb(mu, sigma, beta):
    return np.asarray(mu, dtype=float) + float(beta) * np.asarray(sigma, dtype=float)


# ── AD計算 ────────────────────────────────────────────────────────────────────


def compute_train_knn_distances(X_train_scaled, k):
    n = X_train_scaled.shape[0]
    if n < 2:
        return np.zeros(n)
    effective_k = min(max(1, k), n - 1)
    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    nn.fit(X_train_scaled)
    distances, _ = nn.kneighbors(X_train_scaled)
    return distances[:, 1 : effective_k + 1].mean(axis=1)


def compute_query_knn_distances(X_train_scaled, X_query_scaled, k):
    n = X_train_scaled.shape[0]
    effective_k = min(max(1, k), n)
    nn = NearestNeighbors(n_neighbors=effective_k, metric="euclidean")
    nn.fit(X_train_scaled)
    distances, _ = nn.kneighbors(X_query_scaled)
    return distances.mean(axis=1)


def compute_ad_penalty(ad_ratio, mode: AdMode):
    ad_ratio = np.asarray(ad_ratio, dtype=float)
    if mode == "none":
        return np.ones_like(ad_ratio)
    if mode == "hard":
        return (ad_ratio <= 1.0).astype(float)
    if mode == "soft":
        return np.exp(-(ad_ratio**2))
    raise ValueError(
        f"ad_mode='{mode}' は無効です。'none' / 'hard' / 'soft' のいずれかを指定してください。"
    )


# ── モデル別 mu / sigma 取得（ここが最大の変更点）─────────────────────────────


def _predict_mu_sigma(
    method: str,
    model: Any,
    X_query_scaled: np.ndarray,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    各モデルに応じた予測値 mu と 生のσを返す。

    - GPR / BayesianRidge : return_std=True で直接取得
    - OLS / Ridge / Lasso / ElasticNet / SVR : 訓練残差の RMSE を σ の代理値として使用
    （全クエリ点で同一値 → 相対的な不確かさは AD で表現する設計）
    """
    if method in {"gpr", "blr"}:
        mu, sigma_raw = model.predict(X_query_scaled, return_std=True)

    elif method in {"ols", "ridge", "lasso", "elasticnet", "svr"}:
        mu = model.predict(X_query_scaled)
        y_train_pred = model.predict(X_train_scaled)
        residuals = y_train - y_train_pred
        rmse = float(np.sqrt(np.mean(residuals**2)))
        sigma_raw = np.full(len(mu), max(rmse, _SIGMA_FLOOR))

    else:
        raise ValueError(f"method='{method}' は未対応です。対応: {_VALID_METHODS}")

    return np.asarray(mu, dtype=float), np.asarray(sigma_raw, dtype=float)


# ── ユーティリティ ────────────────────────────────────────────────────────────


def transform_features(X: pd.DataFrame, scaler: Any | None) -> np.ndarray:
    if scaler is None:
        return X.to_numpy(dtype=float)
    return scaler.transform(X)


def attach_known_flags(candidate_df, train_df):
    known_by_cas = pd.Series(False, index=candidate_df.index)
    known_by_smiles = pd.Series(False, index=candidate_df.index)

    if "cas" in candidate_df.columns and "cas" in train_df.columns:
        train_cas = set(train_df["cas"].dropna().astype(str))
        known_by_cas = candidate_df["cas"].astype(str).isin(train_cas)

    if "smiles" in candidate_df.columns and "smiles" in train_df.columns:
        train_smiles = set(train_df["smiles"].dropna().astype(str))
        known_by_smiles = candidate_df["smiles"].astype(str).isin(train_smiles)

    return known_by_cas, known_by_smiles


# ── メイン予測関数 ─────────────────────────────────────────────────────────────


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
    ad_mode: AdMode = "soft",
    sort_by: str = "ei_ad",
) -> tuple[pd.DataFrame, Path]:

    # ── 入力バリデーション（処理前に全チェック）──────────────────────────────
    if method not in _VALID_METHODS:
        raise ValueError(f"method='{method}' は無効です。対応: {_VALID_METHODS}")

    if target not in TARGET:
        raise ValueError(f"target='{target}' は無効です。対応: {TARGET}")

    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level は (0, 1) の範囲で指定してください: {ci_level}")

    # ── モデル・サマリ読み込み ─────────────────────────────────────────────
    estimator = joblib.load(joblib_path)

    if not hasattr(estimator, "named_steps"):
        raise TypeError("estimator は sklearn.pipeline.Pipeline である必要があります。")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # GPR以外は sigma_ キーが存在しないため get() でデフォルト 1.0
    sigma_scale = float(summary["final_best_on_full_data"].get("sigma_", 1.0))

    model = estimator.named_steps[method]
    scaler = estimator.named_steps.get("scaler")  # スケーラーなしのパイプラインにも対応
    descriptors = summary["final_best_combi_on_full_data"]

    # ── 訓練データ準備 ─────────────────────────────────────────────────────
    train_df = train_df.dropna(subset=descriptors + [target]).reset_index(drop=True)

    if target not in train_df.columns:
        raise KeyError(f"train_df に '{target}' 列が見つかりません。")

    current_best_y = float(train_df[target].max())

    # ── sort_by の事前検証 ─────────────────────────────────────────────────
    expected_result_cols = {
        "mu",
        "sigma_raw",
        "sigma_cal",
        "ci_lower",
        "ci_upper",
        "pi",
        "ei",
        "ucb",
        "ad_knn_distance",
        "ad_ratio",
        "in_ad",
        "ad_penalty",
        "ei_ad",
    }
    candidate_cols = (
        set(candidate_df.columns)
        | expected_result_cols
        | {"prediction_status", "known_any"}
    )
    if sort_by not in candidate_cols:
        raise ValueError(f"sort_by='{sort_by}' は有効な列名ではありません。")

    # ── 出力 DataFrame 初期化 ──────────────────────────────────────────────
    output_df = candidate_df.copy()
    output_df["prediction_status"] = "missing_descriptor"

    valid_mask = candidate_df[descriptors].notna().all(axis=1)
    valid_idx = output_df.index[valid_mask]

    known_by_cas, known_by_smiles = attach_known_flags(candidate_df, train_df)
    known_any = known_by_cas | known_by_smiles
    output_df["known_any"] = known_any

    # ── 予測 ───────────────────────────────────────────────────────────────
    X_train = train_df[descriptors].astype(float)
    X_train_scaled = transform_features(X_train, scaler)
    y_train = train_df[target].to_numpy(dtype=float)

    X_query = candidate_df.loc[valid_mask, descriptors].astype(float)
    X_query_scaled = transform_features(X_query, scaler)

    # モデル別 mu / sigma 取得
    mu, sigma_raw = _predict_mu_sigma(
        method, model, X_query_scaled, X_train_scaled, y_train
    )
    if not method == "blr":
        sigma_cal = sigma_raw * sigma_scale
    else:
        sigma_cal = sigma_raw  # BayesianRidge はすでにキャリブレーションされている前提

    # AD計算
    train_knn_dist = compute_train_knn_distances(X_train_scaled, ad_k)
    ad_threshold = float(np.quantile(train_knn_dist, ad_quantile))
    query_knn_dist = compute_query_knn_distances(X_train_scaled, X_query_scaled, ad_k)
    ad_ratio = query_knn_dist / max(ad_threshold, _SIGMA_FLOOR)
    ad_in = ad_ratio <= 1.0
    ad_penalty = compute_ad_penalty(ad_ratio, ad_mode)

    # 獲得関数
    ei = calculate_ei(mu=mu, sigma=sigma_cal, current_best=current_best_y, xi=xi)
    pi = calculate_pi(mu=mu, sigma=sigma_cal, current_best=current_best_y, xi=xi)
    ucb = calculate_ucb(mu=mu, sigma=sigma_cal, beta=beta)
    ei_ad = ei * ad_penalty

    # 既知化合物のスコアをゼロにする
    valid_known = known_any.loc[valid_idx].to_numpy(dtype=bool)
    ei = np.where(valid_known, 0.0, ei)
    pi = np.where(valid_known, 0.0, pi)
    ucb = np.where(valid_known, -np.inf, ucb)
    ei_ad = np.where(valid_known, 0.0, ei_ad)

    # 信頼区間
    z = float(norm.ppf(1.0 - (1.0 - ci_level) / 2.0))
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
            "xi": xi,
            "beta": beta,
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

    # ── ソート・保存 ───────────────────────────────────────────────────────
    if sort_by not in output_df.columns:
        raise ValueError(f"sort_by='{sort_by}' 列が出力 DataFrame に見つかりません。")

    by_cols = (
        [sort_by, "mu"] if "mu" in output_df.columns and sort_by != "mu" else [sort_by]
    )
    output_df = output_df.sort_values(
        by=by_cols,
        ascending=[False] * len(by_cols),
        na_position="last",
    ).reset_index(drop=True)

    today = date.today().strftime("%Y%m%d")  # 関数内で取得（モジュールレベルから移動）
    output_path = Path(__file__).resolve().parent / f"out/{today}_{method}_output.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    return output_df, output_path
