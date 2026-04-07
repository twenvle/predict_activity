import itertools
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

today = date.today().strftime("%Y%m%d")


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

METHOD = {
    "gpr": GaussianProcessRegressor(n_restarts_optimizer=10, random_state=42),
    "ols": LinearRegression(),
    "ridge": Ridge(random_state=42),
    "lasso": Lasso(random_state=42),
    "elasticnet": ElasticNet(random_state=42),
    "pls": PLSRegression(),
    "svr": SVR(),
    "blr": BayesianRidge(),
}

PARAMETER = {
    "alpha",
    "kernel",
    "l1_ratio",
    "n_components",
    "C",
    "epsilon",
    "gamma",
    "alpha_",
    "kernel_",
    "coef_",
    "intercept_",
    "support_vectors_",
    "lambda_",
    "sigma_",
    "x_weights_",
    "x_loadings_",
    "y_loadings_",
}


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def setup_logger(method: str):
    if logger.hasHandlers():
        logger.handlers.clear()

    log_dir = Path("out/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler1 = logging.FileHandler(f"{log_dir}/{today}_train_{method}_info.log")
    file_handler1.setLevel(logging.INFO)

    file_handler2 = logging.FileHandler(f"{log_dir}/{today}_train_{method}_debug.log")
    file_handler2.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler1.setFormatter(formatter)
    file_handler2.setFormatter(formatter)

    logger.addHandler(file_handler1)
    logger.addHandler(file_handler2)


def hyper_parameter(length: int, method: str) -> dict:
    if method == "gpr":
        kernels = [
            # pattern 1: RBF kernel (非常に滑らかな関数を想定)
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF([1.0] * length, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),
            # pattern 2: Matern kernel, nu=2.5 (RBFより少しだけ粗い変化を許容)
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern([1.0] * length, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),
            # pattern 3: Matern kernel, nu=1.5 (さらに粗い変化を許容)
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern([1.0] * length, length_scale_bounds=(1e-2, 1e2), nu=1.5)
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),
        ]
        param_grid = {"gpr__kernel": kernels}
    elif method in {"ols", "blr"}:
        param_grid = {}
    elif method in {"ridge", "lasso"}:
        alpha = np.logspace(-4, 4, 50)
        param_grid = {f"{method}__alpha": alpha}
    elif method == "elasticnet":
        alpha = np.logspace(-4, 4, 50)
        l1_ratio = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        param_grid = {
            "elasticnet__alpha": alpha.tolist(),
            "elasticnet__l1_ratio": l1_ratio.tolist(),
        }
    elif method == "pls":
        n_components = [1, 2, 3, 4, 5, 6, 7]
        param_grid = {"pls__n_components": n_components.tolist()}
    elif method == "svr":
        kernels = ["rbf", "poly"]
        C = [0.1, 1, 10, 100]
        epsilon = [0.01, 0.1, 0.5, 1]
        gamma = ["scale", 0.01, 0.1, 1]
        param_grid = {
            "svr__kernel": kernels,
            "svr__C": C,
            "svr__epsilon": epsilon,
            "svr__gamma": gamma,
        }
    return param_grid


def get_clf(length: int, cv: Any, method: str) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (method, METHOD[method]),
        ]
    )
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=hyper_parameter(length, method),
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit="mae",
    )
    return clf


def fit_row(combi: tuple[str, ...], clf: GridSearchCV, method: str) -> dict:
    row = {
        "combi": combi,
        "inner_cv_mae": float(
            -clf.best_score_
        ),  # neg_mean_absolute_error is negative number, so this code changes "-mae" to "mae".
        "best_estimator": clf.best_estimator_,  # When you use GridSearchCV, you should write this code at first.
    }
    best = clf.best_estimator_.named_steps[method]
    for parameter in PARAMETER:
        if hasattr(best, parameter):
            row[parameter] = getattr(best, parameter)
            if parameter == "kernel_":
                row[parameter] = str(row[parameter])
            if isinstance(row[parameter], np.ndarray):
                row[parameter] = row[parameter].tolist()
            if isinstance(row[parameter], np.floating):
                row[parameter] = float(row[parameter])
    return row


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
    method: str,
) -> dict:
    best = None
    inner_cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    for combi in combinations:
        X_train = X_train_all[list(combi)]
        clf = get_clf(len(combi), inner_cv, method)
        clf.fit(X_train, y_train_all)
        if not np.isfinite(clf.best_score_):
            continue
        row = fit_row(combi, clf, method)
        for key, value in row.items():
            logger.debug(f"{key}: {value}")
        logger.debug(
            f"{combi}: mae -> {row['inner_cv_mae']}, rmse -> {clf.cv_results_['mean_test_rmse'][clf.best_index_]}, r2 -> {clf.cv_results_['mean_test_r2'][clf.best_index_]}"
        )
        if best is None or row["inner_cv_mae"] < best["inner_cv_mae"]:
            best = row
    if best is None:
        raise RuntimeError("inner_cv is failure.")
    for key, value in best.items():
        logger.info(f"best_{key}: {value}")
    return best


def train(
    df: pd.DataFrame,
    method: str,
    descriptors: list[str] = DESCRIPTORS,
    target: str = "yield",
):
    setup_logger(method)
    start_time = time.time()
    X = df[descriptors]
    if target not in TARGET or method not in METHOD.keys():
        raise ValueError(f"{target} or {method} is not exist.")
    y = df[target]

    logger.info(f"target: {target}")
    logger.info(f"method: {method}")
    logger.info(f"len_descriptors: {len(descriptors)}")

    predictions = []
    outer_cv = LeaveOneOut()

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        X_train_all = X.iloc[train_idx]
        y_train_all = y.iloc[train_idx]
        X_test_all = X.iloc[test_idx]
        y_test = float(y.iloc[test_idx].iloc[0])

        corr_abs = X_train_all.corr().abs()
        if method in {"gpr", "ols", "svr", "blr"}:
            combinations = generate_combinations(descriptors, corr_abs)
            logger.info(f"length of combinations: {len(combinations)}")
            best = best_combination(X_train_all, y_train_all, combinations, method)

        elif method in {"ridge", "lasso", "elasticnet", "pls"}:
            best = best_combination(
                X_train_all, y_train_all, [tuple(DESCRIPTORS)], method
            )

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

    if method in {"gpr", "ols", "svr", "blr"}:
        final_corr_abs = X.corr().abs()
        final_combinations = generate_combinations(descriptors, final_corr_abs)
    elif method in {"ridge", "lasso", "elasticnet", "pls"}:
        final_combinations = [tuple[DESCRIPTORS]]
    final_best = best_combination(X, y, final_combinations, method)
    logger.info(final_best)

    pred_path = Path(__file__).resolve().parent / f"out/{today}_predictions.csv"
    summary_path = Path(__file__).resolve().parent / f"out/{today}_summary.json"

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, index=False)

    summary = {
        "n_rows_after_dropna": int(len(df)),
        "n_descriptors": int(len(descriptors)),
        "descriptors": descriptors,
        "n_combinations": int(len(final_combinations)),
        "metrics": metrics,
        "final_best_on_full_data": final_best,
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    final_best.pop("best_estimator", None)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
