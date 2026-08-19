import itertools
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
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
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DESCRIPTORS = [
    "sasa",
    "polar",
    "h_nbo_charge",
    "o_nbo_charge",
    "molecular_volume_A3",
    "dipole_moment_debye",
    "homo_ev",
    "lumo_ev",
    "gap_ev",
    "omega",
    "3and6",
    "4and5",
    "logp",
    "hbd",
    "hba",
]

TARGET = {"yield", "conversion", "selectivity"}
METHOD = {"gpr", "ols", "ridge", "lasso", "elasticnet", "pls", "svr", "blr"}

ATTRIBUTE_TO_LOG = {
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


# Generate logfile to check.
# The type of logfile are debug and information. Basically, you should only need to read info.log.
def setup_logger(file_name: str) -> None:
    global logger

    if logger.hasHandlers():
        logger.handlers.clear()

    log_dir_debug = Path(__file__).resolve().parent.parent / "out/logs/debug"
    log_dir_info = Path(__file__).resolve().parent.parent / "out/logs/info"
    log_dir_debug.mkdir(parents=True, exist_ok=True)
    log_dir_info.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler1 = logging.FileHandler(f"{log_dir_info}/{file_name}_info.log")
    file_handler1.setLevel(logging.INFO)

    file_handler2 = logging.FileHandler(f"{log_dir_debug}/{file_name}_debug.log")
    file_handler2.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler1.setFormatter(formatter)
    file_handler2.setFormatter(formatter)

    logger.addHandler(file_handler1)
    logger.addHandler(file_handler2)


# Select the regression method.
def get_method(method: str):
    methods = {
        # If you set normalize_y=True, GPR will standardize y internally during training and return predictions in the original scale.
        "gpr": GaussianProcessRegressor(
            n_restarts_optimizer=5,
            random_state=42,
            normalize_y=True,
        ),
        "ols": LinearRegression(),
        "ridge": Ridge(random_state=42),
        "lasso": Lasso(random_state=42, max_iter=100000),
        "elasticnet": ElasticNet(random_state=42, max_iter=100000),
        "pls": PLSRegression(),
        "svr": SVR(),
        "blr": BayesianRidge(),
    }
    return methods[method]


# The hyperparameter search space for each regression method.
def hyper_parameter(length: int, method: str, n_samples: int = None) -> dict:
    if method == "gpr":
        kernels = [
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF([1.0] * length, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern([1.0] * length, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1)),
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
        # L1_ratio=0 is excluded because it overlaps with Ridge and tends to produce warnings in ElasticNet.
        l1_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        param_grid = {
            "elasticnet__alpha": alpha,
            "elasticnet__l1_ratio": l1_ratio,
        }

    elif method == "pls":
        if n_samples is None:
            max_comp = min(7, length)
        else:
            # CV内のtraining sizeを考慮して少し保守的に制限
            max_comp = min(7, length, max(1, n_samples - 2))
        n_components = list(range(1, max_comp + 1))
        param_grid = {"pls__n_components": n_components}

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

    else:
        raise ValueError(f"Unsupported method: {method}")

    return param_grid


def get_clf(length: int, cv: Any, method: str, n_samples=None) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (method, get_method(method)),
        ]
    )

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=hyper_parameter(length, method, n_samples=n_samples),
        cv=cv,
        scoring=scoring,
        refit="mae",
        return_train_score=True,
    )
    return clf


def get_outer_cv(
    cv_type: str,
    n_samples: int,
    n_splits: int = 5,
    n_repeats: int = 20,
    random_state: int = 42,
):
    if cv_type == "loo":
        return LeaveOneOut()

    if cv_type == "kfold":
        n_splits = min(n_splits, n_samples)
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if cv_type == "repeated_kfold":
        n_splits = min(n_splits, n_samples)
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        return RepeatedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

    raise ValueError(f"Unsupported cv_type: {cv_type}")


def safe_json(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (list, tuple, dict, str, int, float, bool)) or x is None:
        return x
    return str(x)


def fit_row(combi: tuple[str, ...], clf: GridSearchCV, method: str) -> dict:
    row = {
        "combi": combi,
        "inner_cv_mae": float(-clf.best_score_),
        "best_estimator": clf.best_estimator_,
    }

    best = clf.best_estimator_.named_steps[method]
    for parameter in ATTRIBUTE_TO_LOG:
        if hasattr(best, parameter):
            row[parameter] = safe_json(getattr(best, parameter))

    return row


def has_high_correlation(combi, corr_abs, threshold):
    if len(combi) < 2:
        return False

    for c1, c2 in itertools.combinations(combi, 2):
        if corr_abs.loc[c1, c2] > threshold:
            logger.debug(f"corr_abs > {threshold}: {combi} -> {c1} & {c2}")
            return True
    return False


def generate_combinations(
    descriptors: list, corr_abs: pd.DataFrame, threshold: float, len_descriptors: int
) -> list[tuple[str, ...]]:
    result = []

    for i in range(1, len_descriptors + 1):
        for combi in itertools.combinations(descriptors, i):
            if {"homo_ev", "lumo_ev", "gap_ev"}.issubset(combi):
                continue
            if has_high_correlation(combi, corr_abs, threshold):
                continue
            result.append(combi)

    return result


def best_combination(
    X_train_all: pd.DataFrame,
    y_train_all: pd.Series,
    combinations: list[tuple[str, ...]],
    method: str,
    inner_n_splits: int = 5,
    inner_n_repeats: int = 10,
    random_state: int = 42,
) -> tuple[dict, tuple[str, ...]]:
    best = None
    best_combi = None

    n_samples = len(X_train_all)
    inner_n_splits = min(inner_n_splits, n_samples)
    if inner_n_splits < 2:
        raise ValueError("Not enough samples for inner cross-validation.")

    inner_cv = RepeatedKFold(
        n_splits=inner_n_splits,
        n_repeats=inner_n_repeats,
        random_state=random_state,
    )

    for combi in combinations:
        X_train = X_train_all[list(combi)]
        clf = get_clf(len(combi), inner_cv, method, n_samples=n_samples)
        clf.fit(X_train, y_train_all)

        if not np.isfinite(clf.best_score_):
            continue

        row = fit_row(combi, clf, method)

        for key, value in row.items():
            logger.debug(f"{key}: {value}")

        inner_cv_rmse = -clf.cv_results_["mean_test_rmse"][clf.best_index_]
        inner_cv_r2 = clf.cv_results_["mean_test_r2"][clf.best_index_]
        logger.debug(
            f"{combi}: mae -> {row['inner_cv_mae']}, "
            f"rmse -> {inner_cv_rmse}, r2 -> {inner_cv_r2}"
        )

        if best is None or row["inner_cv_mae"] < best["inner_cv_mae"]:
            best = row
            best_combi = combi

    if best is None or best_combi is None:
        raise RuntimeError("inner_cv is failure.")

    for key, value in best.items():
        logger.info(f"best_{key}: {value}")

    return best, best_combi


def load_checkpoint(checkpoint_path: Path):
    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        predictions = checkpoint.get("predictions", [])
        fold_metrics = checkpoint.get("fold_metrics", [])

        selected_descriptor_counter = Counter(
            checkpoint.get("selected_descriptor_counter", {})
        )

        selected_set_counter = Counter(
            {
                tuple(k.split(",")): v
                for k, v in checkpoint.get("selected_set_counter", {}).items()
            }
        )

        completed_folds = {int(item["fold"]) for item in fold_metrics}

        return (
            predictions,
            fold_metrics,
            selected_descriptor_counter,
            selected_set_counter,
            completed_folds,
        )

    return [], [], Counter(), Counter(), set()


def save_checkpoint(
    predictions,
    fold_metrics,
    selected_descriptor_counter,
    selected_set_counter,
    checkpoint_path: Path,
):
    checkpoint = {
        "predictions": predictions,
        "fold_metrics": fold_metrics,
        "selected_descriptor_counter": dict(selected_descriptor_counter),
        "selected_set_counter": {
            ",".join(k): v for k, v in selected_set_counter.items()
        },
    }

    with checkpoint_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def nested_cv_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    descriptors: list[str],
    method: str,
    checkpoint_path: Path,
    threshold: float = 0.8,
    len_descriptors: int = 4,
    cv_type: str = "loo",
    outer_n_splits: int = 5,
    outer_n_repeats: int = 10,
    inner_n_splits: int = 5,
    inner_n_repeats: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame, dict]:
    if checkpoint_path is not None:
        (
            predictions,
            fold_metrics,
            selected_descriptor_counter,
            selected_set_counter,
            completed_folds,
        ) = load_checkpoint(checkpoint_path)

        logger.info(f"Loaded checkpoint: {len(completed_folds)} folds completed")

    else:
        predictions = []
        fold_metrics = []

        selected_descriptor_counter = Counter()
        selected_set_counter = Counter()

        completed_folds = set()

    outer_cv = get_outer_cv(
        cv_type=cv_type,
        n_samples=len(X),
        n_splits=outer_n_splits,
        n_repeats=outer_n_repeats,
        random_state=random_state,
    )

    splits = list(outer_cv.split(X))

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train_all = X.iloc[train_idx]
        y_train_all = y.iloc[train_idx]
        X_test_all = X.iloc[test_idx]

        if fold_idx in completed_folds:
            logger.info(f"[{cv_type}] fold {fold_idx} skipped (checkpoint)")
            continue

        corr_abs = X_train_all.corr().abs()
        combinations = generate_combinations(
            descriptors,
            corr_abs,
            threshold,
            len_descriptors,
        )

        logger.info(f"[{cv_type}] fold {fold_idx}/{len(splits)}")
        logger.info(f"length of combinations: {len(combinations)}")

        best, _ = best_combination(
            X_train_all,
            y_train_all,
            combinations,
            method,
            inner_n_splits=inner_n_splits,
            inner_n_repeats=inner_n_repeats,
            random_state=random_state,
        )

        combi = list(best["combi"])
        estimator = best["best_estimator"]

        selected_descriptor_counter.update(combi)
        selected_set_counter.update([tuple(combi)])

        y_pred = estimator.predict(X_test_all[combi]).ravel()

        y_true_fold = y.iloc[test_idx].values.astype(float)
        y_pred_fold = np.asarray(y_pred).astype(float)

        if len(y_true_fold) >= 2:
            fold_r2 = float(r2_score(y_true_fold, y_pred_fold))
        else:
            fold_r2 = np.nan

        fold_metric = {
            "fold": fold_idx,
            "n_test_samples": len(test_idx),
            "mae": float(mean_absolute_error(y_true_fold, y_pred_fold)),
            "rmse": float(np.sqrt(mean_squared_error(y_true_fold, y_pred_fold))),
            "r2": fold_r2,
        }

        fold_metrics.append(fold_metric)

        for local_i, idx in enumerate(test_idx):
            y_true_i = float(y.iloc[idx])
            y_pred_i = float(y_pred[local_i])
            predictions.append(
                {
                    "cv_type": cv_type,
                    "fold": fold_idx,
                    "sample_index": int(idx),
                    "selected_descriptor_set": ",".join(combi),
                    "n_descriptors": len(combi),
                    "inner_cv_mae": best["inner_cv_mae"],
                    "y_true": y_true_i,
                    "y_pred": y_pred_i,
                    "absolute_error": abs(y_true_i - y_pred_i),
                }
            )
            logger.info(
                f"pred={y_pred_i:.4f}, true={y_true_i:.4f}, "
                f"abs_err={predictions[-1]['absolute_error']:.4f}"
            )

        logger.info(
            f"[{cv_type}] fold {fold_idx}/{len(splits)} selected={','.join(combi)}\n"
        )

        if checkpoint_path is not None:
            save_checkpoint(
                predictions,
                fold_metrics,
                selected_descriptor_counter,
                selected_set_counter,
                checkpoint_path,
            )

    pred_df = pd.DataFrame(predictions)

    if pred_df.empty:
        raise RuntimeError(
            "No predictions were generated. "
            "Check checkpoint handling and CV configuration."
        )

    metrics = {
        "mae": float(mean_absolute_error(pred_df["y_true"], pred_df["y_pred"])),
        "rmse": float(
            np.sqrt(mean_squared_error(pred_df["y_true"], pred_df["y_pred"]))
        ),
        "r2": float(r2_score(pred_df["y_true"], pred_df["y_pred"])),
    }
    logger.info(f"mae: {metrics['mae']}")
    logger.info(f"rmse: {metrics['rmse']}")
    logger.info(f"r2: {metrics['r2']}\n")

    fold_metrics_df = pd.DataFrame(fold_metrics)

    if fold_metrics_df.empty:
        fold_summary = None

    else:
        fold_summary = {
            "mae_mean": float(fold_metrics_df["mae"].mean()),
            "mae_std": float(fold_metrics_df["mae"].std(ddof=1)),
            "rmse_mean": float(fold_metrics_df["rmse"].mean()),
            "rmse_std": float(fold_metrics_df["rmse"].std(ddof=1)),
            "r2_mean": float(fold_metrics_df["r2"].mean()),
            "r2_std": float(fold_metrics_df["r2"].std(ddof=1)),
        }

    selection_summary = {
        "feature_frequency": {
            key: int(value) for key, value in selected_descriptor_counter.items()
        },
        "selected_set_frequency": {
            ",".join(key): int(value) for key, value in selected_set_counter.items()
        },
    }

    return pred_df, metrics, selection_summary, fold_metrics_df, fold_summary


def permutation_test(
    X: pd.DataFrame,
    y: pd.Series,
    descriptors: list[str],
    method: str,
    checkpoint_path: Path,
    observed_mae: float,
    threshold: float = 0.8,
    len_descriptors: int = 4,
    cv_type: str = "repeated_kfold",
    outer_n_splits: int = 5,
    outer_n_repeats: int = 10,
    inner_n_splits: int = 5,
    inner_n_repeats: int = 10,
    n_permutations: int = 20,
    random_state: int = 42,
) -> dict:
    rng = np.random.default_rng(random_state)
    permutation_mae = []

    for i in range(n_permutations):
        y_permuted = pd.Series(
            rng.permutation(y.values),
            index=y.index,
            name=y.name,
        )

        logger.info(f"permutation test {i + 1}/{n_permutations}")

        perm_checkpoint_path = (
            checkpoint_path.with_name(
                f"{checkpoint_path.stem}_perm_{i + 1}{checkpoint_path.suffix}"
            )
            if checkpoint_path is not None
            else None
        )

        _, perm_metrics, _, _, _ = nested_cv_evaluate(
            X=X,
            y=y_permuted,
            descriptors=descriptors,
            method=method,
            checkpoint_path=perm_checkpoint_path,
            threshold=threshold,
            len_descriptors=len_descriptors,
            cv_type=cv_type,
            outer_n_splits=outer_n_splits,
            outer_n_repeats=outer_n_repeats,
            inner_n_splits=inner_n_splits,
            inner_n_repeats=inner_n_repeats,
            random_state=random_state + i + 1,
        )

        permutation_mae.append(perm_metrics["mae"])

    permutation_mae = np.array(permutation_mae)
    p_value = float(
        (np.sum(permutation_mae <= observed_mae) + 1) / (n_permutations + 1)
    )

    return {
        "n_permutations": int(n_permutations),
        "observed_mae": float(observed_mae),
        "permutation_mae_mean": float(np.mean(permutation_mae)),
        "permutation_mae_std": (
            float(np.std(permutation_mae, ddof=1)) if n_permutations > 1 else 0.0
        ),
        "permutation_mae_min": float(np.min(permutation_mae)),
        "permutation_mae_max": float(np.max(permutation_mae)),
        "p_value_mae": p_value,
        "permutation_mae": permutation_mae.tolist(),
    }


def evaluate_external_test(
    estimator,
    external_df: pd.DataFrame,
    target: str,
    selected_descriptors: list[str],
) -> tuple[pd.DataFrame, dict]:
    X_external = external_df[selected_descriptors]
    y_external = external_df[target]
    y_pred = estimator.predict(X_external).ravel()

    external_pred_df = pd.DataFrame(
        {
            "y_true": y_external.values.astype(float),
            "y_pred": np.asarray(y_pred).astype(float),
        }
    )
    external_pred_df["absolute_error"] = (
        external_pred_df["y_true"] - external_pred_df["y_pred"]
    ).abs()

    external_metrics = {
        "mae": float(
            mean_absolute_error(external_pred_df["y_true"], external_pred_df["y_pred"])
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    external_pred_df["y_true"], external_pred_df["y_pred"]
                )
            )
        ),
        "r2": float(r2_score(external_pred_df["y_true"], external_pred_df["y_pred"])),
    }

    return external_pred_df, external_metrics


def train(
    df: pd.DataFrame,
    method: str,
    name: str,
    descriptors: list[str] = DESCRIPTORS,
    target: str = "yield",
    threshold: float = 0.8,
    len_descriptors: int = 4,
    run: str = "loo",
    outer_n_splits: int = 5,
    outer_n_repeats: int = 10,
    inner_n_splits: int = 5,
    inner_n_repeats: int = 10,
    run_permutation_test: bool = False,
    n_permutations: int = 5,
    external_df: pd.DataFrame = None,
    random_state: int = 42,
):
    if run not in {"loo", "kfold", "repeated_kfold"}:
        raise ValueError(f"Unsupported run type: {run}")
    file_name = (
        f"{os.path.basename(__file__).split('.')[0]}_{method}_{name}_{target}_{run}"
    )
    setup_logger(file_name)
    start_time = time.time()

    checkpoint_path = (
        Path(__file__).resolve().parent.parent
        / "out/logs/checkpoints"
        / f"{file_name}_checkpoint.json"
    )
    checkpoint_path_permutation = (
        Path(__file__).resolve().parent.parent
        / "out/logs/checkpoints"
        / f"{file_name}_permutation_checkpoint.json"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path_permutation.parent.mkdir(parents=True, exist_ok=True)

    if target not in TARGET or method not in METHOD:
        raise ValueError(f"{target} or {method} is not exist.")

    use_cols = descriptors + [target]
    missing_cols = [col for col in use_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in df: {missing_cols}")

    df = df.dropna(subset=use_cols).copy()
    if len(df) < 3:
        raise ValueError(
            "Not enough rows after dropna. At least 3 samples are recommended."
        )

    X = df[descriptors]
    y = df[target]

    logger.info(f"target: {target}")
    logger.info(f"method: {method}")
    logger.info(f"n_rows_after_dropna: {len(df)}")
    logger.info(f"n_descriptors: {len(descriptors)}")
    logger.info(f"max_len_descriptors_in_model: {len_descriptors}\n")

    all_prediction_dfs = []
    all_metrics = {}
    all_selection_summaries = {}

    (
        run_pred_df,
        run_metrics,
        run_selection_summary,
        run_fold_metrics_df,
        run_fold_summary,
    ) = nested_cv_evaluate(
        X=X,
        y=y,
        descriptors=descriptors,
        method=method,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        len_descriptors=len_descriptors,
        cv_type=run,
        outer_n_splits=outer_n_splits,
        outer_n_repeats=outer_n_repeats,
        inner_n_splits=inner_n_splits,
        inner_n_repeats=inner_n_repeats,
        random_state=random_state,
    )
    all_prediction_dfs.append(run_pred_df)
    all_metrics[run] = run_metrics
    all_selection_summaries[run] = run_selection_summary

    pred_df = pd.concat(all_prediction_dfs, ignore_index=True)

    permutation_result = None
    if run_permutation_test:
        if "repeated_kfold" in all_metrics:
            observed_mae = all_metrics["repeated_kfold"]["mae"]
            permutation_cv_type = "repeated_kfold"
        elif "loo" in all_metrics:
            observed_mae = all_metrics["loo"]["mae"]
            permutation_cv_type = "loo"
        else:
            first_key = next(iter(all_metrics))
            observed_mae = all_metrics[first_key]["mae"]
            permutation_cv_type = first_key

        permutation_result = permutation_test(
            X=X,
            y=y,
            descriptors=descriptors,
            method=method,
            checkpoint_path=checkpoint_path_permutation,
            observed_mae=observed_mae,
            threshold=threshold,
            len_descriptors=len_descriptors,
            cv_type=permutation_cv_type,
            outer_n_splits=outer_n_splits,
            outer_n_repeats=outer_n_repeats,
            inner_n_splits=inner_n_splits,
            inner_n_repeats=inner_n_repeats,
            n_permutations=n_permutations,
            random_state=random_state,
        )

    final_corr_abs = X.corr().abs()
    final_combinations = generate_combinations(
        descriptors,
        final_corr_abs,
        threshold,
        len_descriptors,
    )

    final_best, final_best_combi = best_combination(
        X,
        y,
        final_combinations,
        method,
        inner_n_splits=inner_n_splits,
        inner_n_repeats=inner_n_repeats,
        random_state=random_state,
    )
    logger.info(final_best)

    final_selected_descriptors = list(final_best["combi"])

    external_pred_df = None
    external_metrics = None
    if external_df is not None:
        external_use_cols = final_selected_descriptors + [target]
        missing_external_cols = [
            col for col in external_use_cols if col not in external_df.columns
        ]
        if missing_external_cols:
            raise ValueError(f"Missing columns in external_df: {missing_external_cols}")

        external_df = external_df.dropna(subset=external_use_cols).copy()
        external_pred_df, external_metrics = evaluate_external_test(
            estimator=final_best["best_estimator"],
            external_df=external_df,
            target=target,
            selected_descriptors=final_selected_descriptors,
        )
        logger.info(f"external_test_metrics: {external_metrics}")

    out_dir = Path(__file__).resolve().parent.parent / "out/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_pred_path = out_dir / f"{file_name}_predictions.csv"
    json_summary_path = out_dir / f"{file_name}_summary.json"
    pkl_estimator_path = out_dir / f"{file_name}_final_estimator.pkl"
    csv_external_pred_path = out_dir / f"{file_name}_external_predictions.csv"

    pred_df.to_csv(csv_pred_path, index=False)
    if external_pred_df is not None:
        external_pred_df.to_csv(csv_external_pred_path, index=False)

    pkl_estimator = final_best.pop("best_estimator", None)

    summary = {
        "n_rows_after_dropna": int(len(df)),
        "n_descriptors": int(len(descriptors)),
        "descriptors": descriptors,
        "target": target,
        "method": method,
        "threshold": float(threshold),
        "max_len_descriptors_in_model": int(len_descriptors),
        "n_combinations": int(len(final_combinations)),
        "cv_metrics": all_metrics,
        "selection_summaries": all_selection_summaries,
        "permutation_test": permutation_result,
        "external_test_metrics": external_metrics,
        "final_best_on_full_data": final_best,
        "final_best_combi_on_full_data": list(final_best_combi),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "fold_metrics_summary": run_fold_summary,
    }

    joblib.dump(pkl_estimator, pkl_estimator_path)
    with json_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return (
        json_summary_path,
        pkl_estimator_path,
        csv_pred_path,
        (csv_external_pred_path if external_pred_df is not None else None),
    )
