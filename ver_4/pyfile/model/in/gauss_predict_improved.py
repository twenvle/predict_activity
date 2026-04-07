from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors


@dataclass
class ModelBundle:
    model: Any
    scaler: Any | None
    descriptors: list[str]
    current_best_y: float | None = None
    sigma_scale: float = 1.0
    metadata: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gaussian Process 回帰モデルを用いて候補分子を予測し、"
            "EI / PI / UCB と簡易な適用領域 (AD) を付与して CSV 出力します。"
        )
    )
    parser.add_argument("--candidate-csv", type=Path, required=True, help="予測対象 CSV")
    parser.add_argument("--train-csv", type=Path, required=True, help="学習データ CSV")
    parser.add_argument("--model-pkl", type=Path, required=True, help="学習済みモデル joblib / pkl")
    parser.add_argument("--output-csv", type=Path, required=True, help="出力 CSV")
    parser.add_argument(
        "--descriptors",
        nargs="*",
        default=None,
        help=(
            "利用記述子列。通常は pkl 側に feature_names / descriptors を保存しておき、"
            "ここでは省略する運用を推奨します。"
        ),
    )
    parser.add_argument("--target", type=str, default="yield", help="目的変数列名")
    parser.add_argument(
        "--xi",
        type=float,
        default=0.5,
        help=(
            "Expected Improvement の探索係数。目的変数と同じ単位です。"
            "収率[%]なら 0.5〜2.0 程度から試すのが実務的です。"
        ),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="UCB = mu + beta * sigma_cal に使う係数",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="信頼区間列のレベル (例: 0.95)",
    )
    parser.add_argument(
        "--ad-k",
        type=int,
        default=3,
        help="kNN 適用領域で使う近傍数",
    )
    parser.add_argument(
        "--ad-quantile",
        type=float,
        default=0.95,
        help="学習データの自己 kNN 距離分布から AD 閾値を決める分位点",
    )
    parser.add_argument(
        "--ad-penalty",
        choices=["none", "hard", "soft"],
        default="soft",
        help=(
            "EI へ AD をどう反映するか。"
            "hard: AD 外は EI=0、soft: exp(-(ratio^2)) で減衰、none: 反映しない"
        ),
    )
    parser.add_argument(
        "--zero-acquisition-for-known",
        action="store_true",
        help="学習済み分子 (cas / smiles 一致) の獲得関数を 0 にします",
    )
    parser.add_argument(
        "--sort-by",
        choices=["ei", "ei_ad", "ucb", "mu", "sigma_cal"],
        default="ei_ad",
        help="出力 CSV のソート列",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="任意: 実行メタデータ JSON の保存先",
    )
    return parser.parse_args()


def _maybe_extract_pipeline_parts(obj: Any) -> tuple[Any, Any | None]:
    """Pipeline 風オブジェクトなら gpr / scaler を取り出す。"""
    if hasattr(obj, "named_steps"):
        named_steps = getattr(obj, "named_steps")
        if "gpr" in named_steps:
            model = named_steps["gpr"]
            scaler = named_steps.get("scaler")
            return model, scaler
    return obj, None


def load_bundle(model_pkl: Path, descriptors_override: Sequence[str] | None) -> ModelBundle:
    content = joblib.load(model_pkl)

    metadata: dict[str, Any] = {}
    if isinstance(content, dict):
        metadata = content.get("metadata", {}) or {}

        model = (
            content.get("model")
            or content.get("gpr")
            or content.get("best_estimator")
            or content.get("pipeline")
        )
        scaler = content.get("scaler")

        if model is None:
            raise KeyError(
                "pkl 内に model / gpr / best_estimator / pipeline が見つかりません。"
            )

        extracted_model, extracted_scaler = _maybe_extract_pipeline_parts(model)
        model = extracted_model
        if scaler is None:
            scaler = extracted_scaler

        descriptors = list(
            descriptors_override
            or content.get("descriptors")
            or content.get("feature_names")
            or metadata.get("descriptors")
            or []
        )
        current_best_y = content.get("current_best_y", metadata.get("current_best_y"))
        sigma_scale = content.get("sigma_scale", metadata.get("sigma_scale", 1.0))
    else:
        model, scaler = _maybe_extract_pipeline_parts(content)
        descriptors = list(descriptors_override or [])
        current_best_y = None
        sigma_scale = 1.0

    if not descriptors:
        raise ValueError(
            "利用記述子が特定できません。pkl に descriptors / feature_names を保存するか、"
            "--descriptors で明示してください。"
        )

    return ModelBundle(
        model=model,
        scaler=scaler,
        descriptors=descriptors,
        current_best_y=float(current_best_y) if current_best_y is not None else None,
        sigma_scale=float(sigma_scale),
        metadata=metadata,
    )


def validate_required_columns(df: pd.DataFrame, required_columns: Sequence[str], csv_name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_name} に必要な列がありません: {missing}")


def transform_features(X: pd.DataFrame, scaler: Any | None) -> np.ndarray:
    if scaler is None:
        return X.to_numpy(dtype=float)
    return scaler.transform(X)


def calculate_ei(mu: np.ndarray, sigma: np.ndarray, current_best: float, xi: float) -> np.ndarray:
    sigma_safe = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    improvement = np.asarray(mu, dtype=float) - float(current_best) - float(xi)
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(sigma_safe <= 1e-12, 0.0, ei)
    return ei


def calculate_pi(mu: np.ndarray, sigma: np.ndarray, current_best: float, xi: float) -> np.ndarray:
    sigma_safe = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    z = (np.asarray(mu, dtype=float) - float(current_best) - float(xi)) / sigma_safe
    pi = norm.cdf(z)
    pi = np.where(sigma_safe <= 1e-12, 0.0, pi)
    return pi


def calculate_ucb(mu: np.ndarray, sigma: np.ndarray, beta: float) -> np.ndarray:
    return np.asarray(mu, dtype=float) + float(beta) * np.asarray(sigma, dtype=float)


def compute_train_knn_distances(X_train_scaled: np.ndarray, k: int) -> np.ndarray:
    n_samples = X_train_scaled.shape[0]
    if n_samples < 2:
        raise ValueError("AD 計算には少なくとも 2 サンプルの学習データが必要です。")

    effective_k = min(max(1, k), n_samples - 1)
    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    nn.fit(X_train_scaled)
    distances, _ = nn.kneighbors(X_train_scaled, return_distance=True)
    # 0 番目は自分自身なので除く
    return distances[:, 1 : effective_k + 1].mean(axis=1)


def compute_query_knn_distances(X_train_scaled: np.ndarray, X_query_scaled: np.ndarray, k: int) -> np.ndarray:
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


def attach_known_flags(candidate_df: pd.DataFrame, train_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    known_by_cas = pd.Series(False, index=candidate_df.index)
    known_by_smiles = pd.Series(False, index=candidate_df.index)

    if "cas" in candidate_df.columns and "cas" in train_df.columns:
        train_cas = set(train_df["cas"].dropna().astype(str))
        known_by_cas = candidate_df["cas"].astype(str).isin(train_cas)

    if "smiles" in candidate_df.columns and "smiles" in train_df.columns:
        train_smiles = set(train_df["smiles"].dropna().astype(str))
        known_by_smiles = candidate_df["smiles"].astype(str).isin(train_smiles)

    return known_by_cas, known_by_smiles


def main() -> None:
    args = parse_args()

    bundle = load_bundle(args.model_pkl, args.descriptors)

    candidate_df = pd.read_csv(args.candidate_csv)
    train_df_raw = pd.read_csv(args.train_csv)

    validate_required_columns(candidate_df, bundle.descriptors, str(args.candidate_csv))
    validate_required_columns(
        train_df_raw,
        list(bundle.descriptors) + [args.target],
        str(args.train_csv),
    )

    # 学習データは目的変数 + 記述子が揃っているものだけ使う
    train_df = train_df_raw.dropna(subset=list(bundle.descriptors) + [args.target]).reset_index(drop=True)
    if train_df.empty:
        raise ValueError("学習データから有効行を取得できませんでした。")

    current_best_y = (
        float(bundle.current_best_y)
        if bundle.current_best_y is not None
        else float(train_df[args.target].max())
    )

    output_df = candidate_df.copy()
    output_df["prediction_status"] = "missing_descriptor"

    valid_mask = candidate_df[bundle.descriptors].notna().all(axis=1)
    valid_idx = output_df.index[valid_mask]

    # 先に known フラグを付ける
    known_by_cas, known_by_smiles = attach_known_flags(candidate_df, train_df)
    known_any = known_by_cas | known_by_smiles
    output_df["known_by_cas"] = known_by_cas
    output_df["known_by_smiles"] = known_by_smiles
    output_df["known_any"] = known_any

    if len(valid_idx) > 0:
        X_train = train_df[bundle.descriptors].astype(float)
        X_train_scaled = transform_features(X_train, bundle.scaler)

        X_query = candidate_df.loc[valid_mask, bundle.descriptors].astype(float)
        X_query_scaled = transform_features(X_query, bundle.scaler)

        mu, sigma_raw = bundle.model.predict(X_query_scaled, return_std=True)
        sigma_raw = np.asarray(sigma_raw, dtype=float)
        sigma_cal = sigma_raw * float(bundle.sigma_scale)

        # 適用領域 (AD): scaled 空間での kNN 距離
        train_knn_dist = compute_train_knn_distances(X_train_scaled, args.ad_k)
        ad_threshold = float(np.quantile(train_knn_dist, args.ad_quantile))
        query_knn_dist = compute_query_knn_distances(X_train_scaled, X_query_scaled, args.ad_k)
        ad_ratio = query_knn_dist / max(ad_threshold, 1e-12)
        ad_in = ad_ratio <= 1.0
        ad_penalty = compute_ad_penalty(ad_ratio, args.ad_penalty)

        # 獲得関数
        ei = calculate_ei(mu=mu, sigma=sigma_cal, current_best=current_best_y, xi=args.xi)
        pi = calculate_pi(mu=mu, sigma=sigma_cal, current_best=current_best_y, xi=args.xi)
        ucb = calculate_ucb(mu=mu, sigma=sigma_cal, beta=args.beta)
        ei_ad = ei * ad_penalty

        valid_known = known_any.loc[valid_idx].to_numpy(dtype=bool)
        if args.zero_acquisition_for_known:
            ei = np.where(valid_known, 0.0, ei)
            pi = np.where(valid_known, 0.0, pi)
            ei_ad = np.where(valid_known, 0.0, ei_ad)

        alpha = 1.0 - float(args.ci_level)
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
                "xi": float(args.xi),
                "beta": float(args.beta),
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

    sort_col = args.sort_by
    if sort_col not in output_df.columns:
        raise ValueError(f"sort_by='{sort_col}' 列が見つかりません。")

    output_df = output_df.sort_values(
        by=[sort_col, "mu" if "mu" in output_df.columns else sort_col],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)

    if args.metadata_json is not None:
        metadata = {
            "candidate_csv": str(args.candidate_csv),
            "train_csv": str(args.train_csv),
            "model_pkl": str(args.model_pkl),
            "output_csv": str(args.output_csv),
            "descriptors": bundle.descriptors,
            "target": args.target,
            "xi": float(args.xi),
            "beta": float(args.beta),
            "ci_level": float(args.ci_level),
            "ad_k": int(args.ad_k),
            "ad_quantile": float(args.ad_quantile),
            "ad_penalty": args.ad_penalty,
            "sigma_scale": float(bundle.sigma_scale),
            "current_best_y": float(current_best_y),
            "n_candidate_rows": int(len(candidate_df)),
            "n_valid_candidate_rows": int(valid_mask.sum()),
            "n_train_rows": int(len(train_df)),
            "sort_by": sort_col,
            "zero_acquisition_for_known": bool(args.zero_acquisition_for_known),
            "bundle_metadata": bundle.metadata or {},
        }
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        with args.metadata_json.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved prediction CSV to: {args.output_csv}")
    if args.metadata_json is not None:
        print(f"Saved metadata JSON to: {args.metadata_json}")


if __name__ == "__main__":
    main()
