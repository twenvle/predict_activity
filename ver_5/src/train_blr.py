import itertools
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

today = date.today().strftime("%Y%m%d")

log_dir = Path("out/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler1 = logging.FileHandler(f"{log_dir}/{today}_train_blr_info.log")
file_handler1.setLevel(logging.INFO)

file_handler2 = logging.FileHandler(f"{log_dir}/{today}_train_blr_debug.log")
file_handler2.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler1.setFormatter(formatter)
file_handler2.setFormatter(formatter)

logger.addHandler(file_handler1)
logger.addHandler(file_handler2)


DESCRIPTORS = [
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
    "hba",
    "hbd",
]

TARGET = {"yield", "conversion", "selectivity"}


def has_high_correlation(combi, corr_abs):
    if len(combi) < 2:
        return False
    for c1, c2 in itertools.combinations(combi, 2):
        if corr_abs.loc[c1, c2] > 0.8:
            logger.debug(f"corr_abs > 0.8: {combi} -> {c1} & {c2}")
            return True
    return False


def generate_combinations(
    descriptors: list, corr_abs: pd.DataFrame
) -> list[tuple[str, ...]]:
    result = []
    for i in range(1, 5):
        for combi in itertools.combinations(descriptors, i):
            if {"homo_ev", "lumo_ev", "gap_ev"}.issubset(combi):
                continue
            if has_high_correlation(combi, corr_abs):
                continue
            result.append(combi)
    return result


def best_combination(
    X_train_all: pd.DataFrame,
    y_train_all: pd.Series,
    combinations: list[tuple[str, ...]],
) -> dict:
    best = None
    inner_cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    for combi in combinations:
        X_train = X_train_all[list(combi)]
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("blr", BayesianRidge()),
            ]
        )

        scores = cross_validate(
            estimator=pipeline,
            X=X_train,
            y=y_train_all,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=-1,
        )

        mean_mae = float(-scores["test_mae"].mean())
        mean_rmse = float(-scores["test_rmse"].mean())
        mean_r2 = float(scores["test_r2"].mean())

        if not np.isfinite(mean_mae):
            continue
        logger.debug(
            f"{combi}: mae -> {mean_mae}, rmse -> {mean_rmse}, r2 -> {mean_r2}"
        )
        if best is None or mean_mae < best["inner_cv_mae"]:
            best = {
                "combi": combi,
                "inner_cv_mae": mean_mae,
                "pipeline": pipeline,
            }
    if best is None:
        raise RuntimeError("inner_cv is failure.")

    best_combi = list(best["combi"])
    best_estimator = best["pipeline"].fit(X_train_all[best_combi], y_train_all)
    best["best_estimator"] = best_estimator

    logger.info(f"best_combination: {best['combi']}")
    logger.info(f"mae_score: {best['inner_cv_mae']}")
    return best


def train(df: pd.DataFrame, descriptors: list = DESCRIPTORS, target: str = "yield"):
    start_time = time.time()
    X = df[descriptors]
    if target not in TARGET:
        raise ValueError(f"{target} is not exist.")
    y = df[target]

    logger.info(f"len_descriptors: {len(descriptors)}")

    predictions = []
    outer_cv = LeaveOneOut()

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        X_train_all = X.iloc[train_idx]
        y_train_all = y.iloc[train_idx]
        X_test_all = X.iloc[test_idx]
        y_test = float(y.iloc[test_idx].iloc[0])

        corr_abs = X_train_all.corr().abs()
        combinations = generate_combinations(descriptors, corr_abs)
        logger.info(f"length of combinations: {len(combinations)}")

        best = best_combination(X_train_all, y_train_all, combinations)
        combi = list(best["combi"])
        estimator = best["best_estimator"]
        y_pred = float(estimator.predict(X_test_all[combi])[0])

        predictions.append(
            {
                "fold": fold_idx,
                "selected_descriptor_set": "|".join(combi),
                "n_descriptors": len(combi),
                "inner_cv_mae": best["inner_cv_mae"],
                "y_true": y_test,
                "y_pred": y_pred,
                "absolute_error": abs(y_test - y_pred),
            }
        )
        logger.info(
            f"[fold {fold_idx}/{len(df)}] {predictions[-1]['selected_descriptor_set']} -> "
        )
        logger.info(
            f"pred={y_pred:.4f}, true={y_test:.4f}, abs_err={predictions[-1]['absolute_error']:.4f}"
        )

    pred_df = pd.DataFrame(predictions)
    metrics = {
        "mae_loo": float(mean_absolute_error(pred_df["y_true"], pred_df["y_pred"])),
        "rmse_loo": float(
            np.sqrt(mean_squared_error(pred_df["y_true"], pred_df["y_pred"]))
        ),
        "r2_loo": float(r2_score(pred_df["y_true"], pred_df["y_pred"])),
    }
    logger.info(f"mae_loo: {metrics['mae_loo']}")
    logger.info(f"rmse_loo: {metrics['rmse_loo']}")
    logger.info(f"r2_loo: {metrics['r2_loo']}")

    final_corr_abs = X.corr().abs()
    final_combinations = generate_combinations(descriptors, final_corr_abs)
    final_best = best_combination(X, y, final_combinations)
    final_summary = {
        "selected_descriptor_set": "|".join(final_best["combi"]),
        "n_descriptors": len(final_best["combi"]),
        "inner_cv_mae": final_best["inner_cv_mae"],
    }
    logger.info(f"selected_descriptor_set: {final_summary['selected_descriptor_set']}")
    logger.info(f"n_descriptors: {final_summary['n_descriptors']}")
    logger.info(f"inner_cv_mae: {final_summary['inner_cv_mae']}")

    pred_path = Path(__file__).resolve().parent / f"out/{today}_blr_predictions.csv"
    summary_path = Path(__file__).resolve().parent / f"out/{today}_blr_summary.json"

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, index=False)

    summary = {
        "n_rows_after_dropna": int(len(df)),
        "n_descriptors": int(len(descriptors)),
        "descriptors": descriptors,
        "n_combinations": int(len(final_combinations)),
        "metrics": metrics,
        "final_best_on_full_data": final_summary,
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
