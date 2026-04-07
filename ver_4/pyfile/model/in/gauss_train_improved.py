from __future__ import annotations

import argparse
import itertools
import json
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
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
            "Gaussian Process Regression を用いて、CSV 中の数値記述子から "
            "最大指定数までの全組合せを網羅評価します。"
        )
    )
    parser.add_argument(
        "--input-csv", type=Path, required=True, help="入力 CSV ファイル"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/descriptors"),
        help="結果 CSV / JSON の保存先",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="gaussian_process_all_combinations",
        help="出力ファイル名の接頭辞",
    )
    parser.add_argument("--target", type=str, default="yield", help="目的変数の列名")
    parser.add_argument(
        "--max-descriptors",
        type=int,
        default=4,
        help="1 モデルで使う記述子数の上限",
    )
    parser.add_argument(
        "--exclude-columns",
        nargs="*",
        default=DEFAULT_EXCLUDE_COLUMNS,
        help="記述子候補から除外する列名",
    )
    parser.add_argument(
        "--inner-splits",
        type=int,
        default=5,
        help="内側 CV の分割数",
    )
    parser.add_argument(
        "--inner-repeats",
        type=int,
        default=10,
        help="内側 CV の反復回数 (1 なら通常 KFold)",
    )
    parser.add_argument(
        "--n-restarts-optimizer",
        type=int,
        default=10,
        help="GaussianProcessRegressor の optimizer 再スタート回数",
    )
    parser.add_argument(
        "--gridsearch-jobs",
        type=int,
        default=-1,
        help="GridSearchCV の n_jobs",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="乱数シード",
    )
    parser.add_argument(
        "--normalize-y",
        action="store_true",
        help="GaussianProcessRegressor(normalize_y=True) を有効化する",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=None,
        help=(
            "この値を指定すると、組合せ内に |相関係数| が閾値超えのペアがある場合は "
            "その組合せをスキップします"
        ),
    )
    parser.add_argument(
        "--skip-gap-triplets",
        action="store_true",
        help=("homo_ev, lumo_ev, gap_ev を同時に含む組合せをスキップします"),
    )
    parser.add_argument(
        "--save-partial-every",
        type=int,
        default=50,
        help="途中結果を何組合せごとに保存するか",
    )
    return parser.parse_args()


def infer_descriptor_columns(
    df: pd.DataFrame,
    target: str,
    exclude_columns: Sequence[str],
) -> list[str]:
    if target not in df.columns:
        raise ValueError(f"目的変数列 '{target}' が CSV に見つかりません。")

    excluded = set(exclude_columns) | {target}
    numeric_columns = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
    ]
    descriptors = [col for col in numeric_columns if col not in excluded]

    if not descriptors:
        raise ValueError("利用可能な数値記述子列が見つかりません。")

    return descriptors


def build_kernels(n_features: int) -> list:
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


def build_inner_cv(
    n_train_samples: int,
    n_splits: int,
    n_repeats: int,
    random_state: int,
):
    effective_splits = min(n_splits, n_train_samples)
    if effective_splits < 2:
        raise ValueError("内側 CV の分割数が 2 未満です。")

    if n_repeats <= 1:
        return KFold(
            n_splits=effective_splits,
            shuffle=True,
            random_state=random_state,
        )

    return RepeatedKFold(
        n_splits=effective_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )


def has_high_correlation(
    combo: Sequence[str],
    abs_corr: pd.DataFrame,
    threshold: float | None,
) -> bool:
    if threshold is None or len(combo) < 2:
        return False

    for col1, col2 in itertools.combinations(combo, 2):
        if abs_corr.loc[col1, col2] > threshold:
            return True
    return False


def make_search(
    n_features: int,
    n_train_samples: int,
    args: argparse.Namespace,
) -> GridSearchCV:
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

    inner_cv = build_inner_cv(
        n_train_samples=n_train_samples,
        n_splits=args.inner_splits,
        n_repeats=args.inner_repeats,
        random_state=args.random_state,
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid={"gpr__kernel": build_kernels(n_features)},
        cv=inner_cv,
        scoring="neg_mean_absolute_error",
        n_jobs=args.gridsearch_jobs,
        error_score=np.nan,
        refit=True,
    )


def count_relevant_warnings(caught_warnings: Sequence[warnings.WarningMessage]) -> int:
    return sum(
        issubclass(w.category, (ConvergenceWarning, FitFailedWarning))
        for w in caught_warnings
    )


def fit_with_warning_capture(search, X_train, y_train):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        warnings.simplefilter("always", FitFailedWarning)
        search.fit(X_train, y_train)
    return search, count_relevant_warnings(caught)


def evaluate_combination(
    df: pd.DataFrame,
    target: str,
    combo: Sequence[str],
    args: argparse.Namespace,
) -> dict:
    X = df[list(combo)]
    y = df[target]

    outer_cv = LeaveOneOut()
    y_true: list[float] = []
    y_pred: list[float] = []
    outer_kernels: list[str] = []
    warning_count = 0

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        search = make_search(
            n_features=len(combo),
            n_train_samples=len(X_train),
            args=args,
        )
        search, fold_warning_count = fit_with_warning_capture(search, X_train, y_train)
        warning_count += fold_warning_count

        if not np.isfinite(search.best_score_):
            raise RuntimeError(f"組合せ {combo} の内側 CV がすべて失敗しました。")

        best_estimator = search.best_estimator_
        outer_kernels.append(str(best_estimator.named_steps["gpr"].kernel_))
        y_pred.append(float(best_estimator.predict(X_test)[0]))
        y_true.append(float(y_test.iloc[0]))

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    final_search = make_search(
        n_features=len(combo),
        n_train_samples=len(X),
        args=args,
    )
    final_search, final_warning_count = fit_with_warning_capture(final_search, X, y)
    warning_count += final_warning_count

    if not np.isfinite(final_search.best_score_):
        final_kernel = "fit_failed"
        final_inner_cv_mae = np.nan
    else:
        final_kernel = str(final_search.best_estimator_.named_steps["gpr"].kernel_)
        final_inner_cv_mae = float(-final_search.best_score_)

    kernel_counter = Counter(outer_kernels)
    outer_kernel_mode, outer_kernel_mode_count = kernel_counter.most_common(1)[0]

    return {
        "descriptor_set": "|".join(combo),
        "n_descriptors": len(combo),
        "n_samples": len(df),
        "rmse_loo": rmse,
        "mae_loo": mae,
        "r2_loo": r2,
        "outer_kernel_mode": outer_kernel_mode,
        "outer_kernel_mode_count": outer_kernel_mode_count,
        "outer_kernel_nunique": len(kernel_counter),
        "final_kernel_full_data": final_kernel,
        "final_inner_cv_mae": final_inner_cv_mae,
        "warning_count": warning_count,
    }


def generate_combinations(
    descriptors: Sequence[str],
    abs_corr: pd.DataFrame,
    max_descriptors: int,
    corr_threshold: float | None,
    skip_gap_triplets: bool,
) -> list[tuple[str, ...]]:
    combinations_to_use: list[tuple[str, ...]] = []

    for size in range(1, max_descriptors + 1):
        for combo in itertools.combinations(descriptors, size):
            combo_set = set(combo)
            if skip_gap_triplets and {"homo_ev", "lumo_ev", "gap_ev"}.issubset(
                combo_set
            ):
                continue
            if has_high_correlation(combo, abs_corr, corr_threshold):
                continue
            combinations_to_use.append(combo)

    return combinations_to_use


def main() -> None:
    args = parse_args()

    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.input_csv)
    descriptors = infer_descriptor_columns(
        raw_df,
        target=args.target,
        exclude_columns=args.exclude_columns,
    )

    required_columns = [args.target] + descriptors
    df = raw_df.dropna(subset=required_columns).reset_index(drop=True)
    dropped_rows = len(raw_df) - len(df)

    if df.empty:
        raise ValueError("欠損除去後のデータが 0 行になりました。")

    max_descriptors = min(args.max_descriptors, len(descriptors))
    abs_corr = df[descriptors].corr().abs()

    combinations_to_use = generate_combinations(
        descriptors=descriptors,
        abs_corr=abs_corr,
        max_descriptors=max_descriptors,
        corr_threshold=args.corr_threshold,
        skip_gap_triplets=args.skip_gap_triplets,
    )

    if not combinations_to_use:
        raise ValueError("評価対象の記述子組合せが 1 つもありません。")

    print(f"input_csv: {args.input_csv}")
    print(f"n_rows_raw: {len(raw_df)}")
    print(f"n_rows_after_dropna: {len(df)}")
    print(f"dropped_rows: {dropped_rows}")
    print(f"target: {args.target}")
    print(f"descriptors ({len(descriptors)}): {descriptors}")
    print(f"max_descriptors: {max_descriptors}")
    print(f"n_combinations: {len(combinations_to_use)}")

    results: list[dict] = []
    partial_path = args.output_dir / f"{args.run_name}_partial.csv"
    final_path = args.output_dir / f"{args.run_name}.csv"
    meta_path = args.output_dir / f"{args.run_name}_metadata.json"

    for idx, combo in enumerate(combinations_to_use, start=1):
        row = evaluate_combination(df=df, target=args.target, combo=combo, args=args)
        results.append(row)

        print(
            f"[{idx}/{len(combinations_to_use)}] {row['descriptor_set']} -> "
            f"MAE={row['mae_loo']:.4f}, RMSE={row['rmse_loo']:.4f}, R2={row['r2_loo']:.4f}"
        )

        if idx % args.save_partial_every == 0 or idx == len(combinations_to_use):
            partial_df = pd.DataFrame(results).sort_values(
                ["mae_loo", "rmse_loo", "r2_loo"],
                ascending=[True, True, False],
            )
            partial_df.to_csv(partial_path, index=False)

    results_df = pd.DataFrame(results).sort_values(
        ["mae_loo", "rmse_loo", "r2_loo"],
        ascending=[True, True, False],
    )
    results_df.to_csv(final_path, index=False)

    metadata = {
        "input_csv": str(args.input_csv),
        "output_csv": str(final_path),
        "partial_csv": str(partial_path),
        "target": args.target,
        "exclude_columns": list(args.exclude_columns),
        "descriptors": descriptors,
        "n_rows_raw": int(len(raw_df)),
        "n_rows_after_dropna": int(len(df)),
        "dropped_rows": int(dropped_rows),
        "max_descriptors": int(max_descriptors),
        "n_combinations": int(len(combinations_to_use)),
        "inner_splits": int(args.inner_splits),
        "inner_repeats": int(args.inner_repeats),
        "n_restarts_optimizer": int(args.n_restarts_optimizer),
        "gridsearch_jobs": int(args.gridsearch_jobs),
        "normalize_y": bool(args.normalize_y),
        "corr_threshold": args.corr_threshold,
        "skip_gap_triplets": bool(args.skip_gap_triplets),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "top_10_by_mae": results_df.head(10).to_dict(orient="records"),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nTop 10 combinations by MAE:")
    print(results_df.head(10).to_string(index=False))
    print(f"\nSaved final results to: {final_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
