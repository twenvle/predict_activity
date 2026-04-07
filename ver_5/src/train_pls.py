import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

today = date.today().strftime("%Y%m%d")

log_dir = Path("out/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler1 = logging.FileHandler(f"{log_dir}/{today}_train_pls_info.log")
file_handler1.setLevel(logging.INFO)

file_handler2 = logging.FileHandler(f"{log_dir}/{today}_train_pls_debug.log")
file_handler2.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler1.setFormatter(formatter)
file_handler2.setFormatter(formatter)

logger.addHandler(file_handler1)
logger.addHandler(file_handler2)

# Gap_ev overlaps in meaning with homo_ev and lumo_ev.
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


def hyper_parameter() -> dict:
    n_components = [1, 2, 3, 4, 5, 6, 7]
    param_grid = {"pls__n_components": n_components.tolist()}
    return param_grid


def get_clf(cv: Any) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(random_state=42)),
        ]
    )
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=hyper_parameter(),
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit="mae",
    )
    return clf


def inner_cv(
    X_train_all: pd.DataFrame, y_train_all: pd.Series, descriptors: list[str]
) -> dict[str, Any]:
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

    X_train = X_train_all[descriptors]
    clf = get_clf(cv)
    clf.fit(X_train, y_train_all)
    if not np.isfinite(clf.best_score_):
        raise RuntimeError("inner_cv is failure.")
    best_model = clf.best_estimator_.named_steps["pls"]
    row = {
        "coef": best_model.coef_.tolist(),
        "intercept": float(best_model.intercept_),
        "inner_cv_mae": float(
            -clf.best_score_
        ),  # neg_mean_absolute_error is negative number, so this code changes "-mae" to "mae".
        "best_estimator": clf.best_estimator_,  # This code get the best model.
    }
    logger.debug(f"{descriptors}")
    logger.debug(f"{row['coef']}")
    logger.debug(f"{row['intercept']}")
    logger.debug(
        f"mae -> {row['inner_cv_mae']}, rmse -> {clf.cv_results_['mean_test_rmse'][clf.best_index_]}, r2 -> {clf.cv_results_['mean_test_r2'][clf.best_index_]}"
    )
    return row


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

        best = inner_cv(X_train_all, y_train_all, descriptors)
        estimator = best["best_estimator"]
        y_pred = float(estimator.predict(X_test_all[descriptors])[0])

        predictions.append(
            {
                "fold": fold_idx,
                "n_descriptors": len(descriptors),
                "inner_cv_mae": best["inner_cv_mae"],
                "y_true": y_test,
                "y_pred": y_pred,
                "absolute_error": abs(y_test - y_pred),
            }
        )
        logger.info(f"[fold {fold_idx}/{len(df)}] -> ")
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

    final_best = inner_cv(X, y, descriptors)
    final_summary = {
        "coef": final_best["coef"],
        "intercept": final_best["intercept"],
        "inner_cv_mae": final_best["inner_cv_mae"],
    }
    logger.info(f"descriptors: {descriptors}")
    logger.info(f"final_coef: {final_summary['coef']}")
    logger.info(f"final_intercept: {final_summary['intercept']}")
    logger.info(f"final_inner_cv_mae: {final_summary['inner_cv_mae']}")

    pred_path = Path(__file__).resolve().parent / f"out/{today}_pls_predictions.csv"
    summary_path = Path(__file__).resolve().parent / f"out/{today}_pls_summary.json"

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, index=False)

    summary = {
        "n_rows_after_dropna": int(len(df)),
        "n_descriptors": int(len(descriptors)),
        "descriptors": descriptors,
        "metrics": metrics,
        "final_best_on_full_data": final_summary,
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
