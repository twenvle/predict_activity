from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_EXCLUDE_COLUMNS = ["cas", "smiles"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "記述子組合せの探索自体を outer CV の内側に入れて、"
            "モデル選択手順全体の汎化性能を見積もります。"
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("out/selection_nested"))
    parser.add_argument("--run-name", type=str, default="gaussian_process_nested_selection")
    parser.add_argument("--target", type=str, default="yield")
    parser.add_argument("--max-descriptors", type=int, default=4)
    parser.add_argument("--exclude-columns", nargs="*", default=DEFAULT_EXCLUDE_COLUMNS)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--inner-repeats", type=int, default=10)
    parser.add_argument("--n-restarts-optimizer", type=int, default=10)
    parser.add_argument("--gridsearch-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--normalize-y", action="store_true")
    parser.add_argument("--corr-threshold", type=float, default=None)
    parser.add_argument("--skip-gap-triplets", action="store_true")
    return parser.parse_args()


def infer_descriptors(df: pd.DataFrame, target: str, exclude_columns: list[str]) -> list[str]:
    excluded = set(exclude_columns) | {target}
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    descriptors = [c for c in numeric_cols if c not in excluded]
    if not descriptors:
        raise ValueError("利用可能な数値記述子が見つかりません。")
    return descriptors


def build_kernels(n_features: int):
    return [
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF([1.0] * n_features, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern([1.0] * n_features, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern([1.0] * n_features, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),
    ]


def build_inner_cv(n_samples: int, n_splits: int, n_repeats: int, random_state: int):
    effective_splits = min(n_splits, n_samples)
    if effective_splits < 2:
        raise ValueError("内側 CV の分割数が 2 未満です。")
    if n_repeats <= 1:
        return KFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
    return RepeatedKFold(
        n_splits=effective_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )


def make_search(n_features: int, n_train_samples: int, args: argparse.Namespace) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gpr",
                GaussianProcessRegressor(
                    n_restarts_optimizer=args.n_restarts_optimizer,
                    random_state=args.random_state,
                    normalize_y=args.normalize_y,
                ),
            ),
        ]
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid={"gpr__kernel": build_kernels(n_features)},
        cv=build_inner_cv(n_train_samples, args.inner_splits, args.inner_repeats, args.random_state),
        scoring="neg_mean_absolute_error",
        n_jobs=args.gridsearch_jobs,
        refit=True,
        error_score=np.nan,
    )


def has_high_correlation(combo: tuple[str, ...], abs_corr: pd.DataFrame, threshold: float | None) -> bool:
    if threshold is None or len(combo) < 2:
        return False
    for c1, c2 in itertools.combinations(combo, 2):
        if abs_corr.loc[c1, c2] > threshold:
            return True
    return False


def generate_combinations(descriptors: list[str], abs_corr: pd.DataFrame, args: argparse.Namespace) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    for size in range(1, min(args.max_descriptors, len(descriptors)) + 1):
        for combo in itertools.combinations(descriptors, size):
            if args.skip_gap_triplets and {"homo_ev", "lumo_ev", "gap_ev"}.issubset(combo):
                continue
            if has_high_correlation(combo, abs_corr, args.corr_threshold):
                continue
            result.append(combo)
    return result


def select_best_combo(
    X_train_all: pd.DataFrame,
    y_train_all: pd.Series,
    combinations_to_use: list[tuple[str, ...]],
    args: argparse.Namespace,
) -> dict:
    best = None
    for combo in combinations_to_use:
        X_train = X_train_all[list(combo)]
        search = make_search(len(combo), len(X_train), args)
        search.fit(X_train, y_train_all)
        if not np.isfinite(search.best_score_):
            continue
        row = {
            "combo": combo,
            "inner_cv_mae": float(-search.best_score_),
            "best_estimator": search.best_estimator_,
            "best_kernel": str(search.best_estimator_.named_steps["gpr"].kernel_),
        }
        if best is None or row["inner_cv_mae"] < best["inner_cv_mae"]:
            best = row
    if best is None:
        raise RuntimeError("すべての組合せで inner CV が失敗しました。")
    return best


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    raw_df = pd.read_csv(args.input_csv)
    descriptors = infer_descriptors(raw_df, args.target, args.exclude_columns)
    df = raw_df.dropna(subset=[args.target] + descriptors).reset_index(drop=True)
    abs_corr = df[descriptors].corr().abs()
    combinations_to_use = generate_combinations(descriptors, abs_corr, args)

    X_all = df[descriptors]
    y_all = df[args.target]

    predictions = []
    outer_cv = LeaveOneOut()

    print(f"n_rows_raw: {len(raw_df)}")
    print(f"n_rows_after_dropna: {len(df)}")
    print(f"n_descriptors: {len(descriptors)}")
    print(f"n_combinations: {len(combinations_to_use)}")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_all), start=1):
        X_train_all = X_all.iloc[train_idx]
        y_train_all = y_all.iloc[train_idx]
        X_test_all = X_all.iloc[test_idx]
        y_test = float(y_all.iloc[test_idx].iloc[0])

        best = select_best_combo(X_train_all, y_train_all, combinations_to_use, args)
        combo = list(best["combo"])
        estimator = best["best_estimator"]
        y_pred = float(estimator.predict(X_test_all[combo])[0])

        predictions.append(
            {
                "fold": fold_idx,
                "selected_descriptor_set": "|".join(combo),
                "n_descriptors": len(combo),
                "selected_kernel": best["best_kernel"],
                "inner_cv_mae": best["inner_cv_mae"],
                "y_true": y_test,
                "y_pred": y_pred,
                "absolute_error": abs(y_test - y_pred),
            }
        )
        print(
            f"[fold {fold_idx}/{len(df)}] {predictions[-1]['selected_descriptor_set']} -> "
            f"pred={y_pred:.4f}, true={y_test:.4f}, abs_err={predictions[-1]['absolute_error']:.4f}"
        )

    pred_df = pd.DataFrame(predictions)
    metrics = {
        "mae_loo": float(mean_absolute_error(pred_df["y_true"], pred_df["y_pred"])),
        "rmse_loo": float(np.sqrt(mean_squared_error(pred_df["y_true"], pred_df["y_pred"]))),
        "r2_loo": float(r2_score(pred_df["y_true"], pred_df["y_pred"])),
    }

    final_best = select_best_combo(X_all, y_all, combinations_to_use, args)
    final_summary = {
        "selected_descriptor_set": "|".join(final_best["combo"]),
        "n_descriptors": len(final_best["combo"]),
        "selected_kernel": final_best["best_kernel"],
        "inner_cv_mae": final_best["inner_cv_mae"],
    }

    pred_path = args.output_dir / f"{args.run_name}_predictions.csv"
    summary_path = args.output_dir / f"{args.run_name}_summary.json"

    pred_df.to_csv(pred_path, index=False)

    summary = {
        "input_csv": str(args.input_csv),
        "n_rows_raw": int(len(raw_df)),
        "n_rows_after_dropna": int(len(df)),
        "n_descriptors": int(len(descriptors)),
        "descriptors": descriptors,
        "n_combinations": int(len(combinations_to_use)),
        "metrics": metrics,
        "final_best_on_full_data": final_summary,
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSelection-procedure performance:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nBest combo on full data:")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    print(f"\nSaved predictions to: {pred_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
