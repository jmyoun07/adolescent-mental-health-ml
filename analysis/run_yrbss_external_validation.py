from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".mplconfig"))

from run_lit_reduced_ml import (  # noqa: E402
    fit_predict_proba,
    normalize_params,
    stratified_limit,
)
from run_protocol_ml import (  # noqa: E402
    PRIMARY_OUTCOMES,
    RANDOM_SEED,
    calibration_slope_intercept,
    feature_importance,
    metric_block,
    sample_model,
    threshold_table,
)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


KYRBS_DATA = PROJECT / "data" / "processed" / "kyrbs_2020_2025_lit_reduced_dataset.csv"
YRBSS_DATA = PROJECT / "data" / "external_yrbss" / "2023_national" / "yrbs2023_national_raw.csv"
OUT_DIR = PROJECT / "outputs" / "external_validation_yrbss_2023"

TARGET_COLS = {
    "M_SAD": "target_depressed_mood",
    "M_SUI_CON": "target_suicidal_ideation",
    "M_SUI_PLN": "target_suicide_plan",
    "M_SUI_ATT": "target_suicide_attempt_secondary",
}

YRBSS_TARGETS = {
    "target_depressed_mood": "QN26",
    "target_suicidal_ideation": "QN27",
    "target_suicide_plan": "QN28",
    "target_suicide_attempt_secondary": "QN29",
}

BASE_FEATURES = [
    "sex",
    "age_years",
    "grade_9_12",
    "academic_achievement_3",
    "body_image_5",
    "bmi",
    "physical_activity_days",
    "physical_activity_5plus",
    "breakfast_days",
    "trying_to_lose_weight",
    "current_alcohol",
    "binge_alcohol_any",
    "lifetime_alcohol_experience",
    "any_current_tobacco",
    "lifetime_tobacco_any",
    "sexual_experience",
    "sleep_hours",
    "sleep_8plus",
    "digital_high_use",
]

CATEGORICAL_FEATURES = ["sex", "grade_9_12"]


def ensure_dirs() -> None:
    for sub in ["", "models", "feature_importance", "probabilities", "trials", "tables"]:
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def make_preprocessor(categorical: list[str], numeric: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("num", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(("cat", categorical_pipe, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace({"": np.nan, ".": np.nan}), errors="coerce")


def binary_yes1_no2(s: pd.Series) -> pd.Series:
    x = num(s)
    return pd.Series(np.where(x == 1, 1.0, np.where(x == 2, 0.0, np.nan)), index=s.index)


def binary_from_bool(condition: pd.Series, known: pd.Series) -> pd.Series:
    return pd.Series(np.where(known, condition.astype(float), np.nan), index=known.index)


def category_code_string(s: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype=object)
    valid = s.notna()
    out.loc[valid] = s.loc[valid].astype(int).astype(str)
    return out


def code_minus_one_days(s: pd.Series, max_code: int = 8) -> pd.Series:
    x = num(s)
    days = x - 1
    return days.where(x.between(1, max_code), np.nan)


def kyrbs_shared_dataset() -> pd.DataFrame:
    usecols = [
        "YEAR",
        "SEX",
        "AGE",
        "GRADE",
        "academic_achievement_3",
        "PR_BI",
        "BMI",
        "PA_TOT",
        "F_BR",
        "WC_MN",
        "current_alcohol",
        "heavy_drunk_any",
        "lifetime_alcohol_experience",
        "any_current_tobacco",
        "lifetime_tobacco_any",
        "sexual_experience",
        "avg_sleep_hours",
        "smartphone_avg_daily_minutes",
        *TARGET_COLS.values(),
    ]
    df = pd.read_csv(KYRBS_DATA, usecols=usecols, low_memory=False)
    df = df[df["YEAR"].between(2020, 2024)].copy()
    df = df[num(df["GRADE"]).between(3, 6)].copy()

    out = pd.DataFrame(index=df.index)
    out["source"] = "KYRBS"
    out["YEAR"] = num(df["YEAR"])
    sex = pd.Series(np.nan, index=df.index, dtype=object)
    sex[num(df["SEX"]).eq(1)] = "male"
    sex[num(df["SEX"]).eq(2)] = "female"
    out["sex"] = sex
    out["age_years"] = num(df["AGE"])
    out["grade_9_12"] = category_code_string(num(df["GRADE"]) + 6)
    out["academic_achievement_3"] = num(df["academic_achievement_3"])
    out["body_image_5"] = num(df["PR_BI"]).where(num(df["PR_BI"]).between(1, 5), np.nan)
    out["bmi"] = num(df["BMI"]).where(num(df["BMI"]).between(10, 60), np.nan)
    out["physical_activity_days"] = code_minus_one_days(df["PA_TOT"])
    out["physical_activity_5plus"] = binary_from_bool(out["physical_activity_days"] >= 5, out["physical_activity_days"].notna())
    out["breakfast_days"] = code_minus_one_days(df["F_BR"])
    out["trying_to_lose_weight"] = binary_from_bool(num(df["WC_MN"]).eq(2), num(df["WC_MN"]).between(1, 4))
    out["current_alcohol"] = num(df["current_alcohol"])
    out["binge_alcohol_any"] = num(df["heavy_drunk_any"])
    out["lifetime_alcohol_experience"] = num(df["lifetime_alcohol_experience"])
    out["any_current_tobacco"] = num(df["any_current_tobacco"])
    out["lifetime_tobacco_any"] = num(df["lifetime_tobacco_any"])
    out["sexual_experience"] = num(df["sexual_experience"])
    out["sleep_hours"] = num(df["avg_sleep_hours"]).where(num(df["avg_sleep_hours"]).between(2, 16), np.nan)
    out["sleep_8plus"] = binary_from_bool(out["sleep_hours"] >= 8, out["sleep_hours"].notna())
    out["digital_high_use"] = binary_from_bool(num(df["smartphone_avg_daily_minutes"]) >= 240, num(df["smartphone_avg_daily_minutes"]).notna())
    out["depressed_mood_for_suicide_models"] = num(df["target_depressed_mood"])

    for target in TARGET_COLS.values():
        out[target] = num(df[target])
    out["sample_weight"] = np.nan
    return out


def yrbss_shared_dataset() -> pd.DataFrame:
    yr = pd.read_csv(YRBSS_DATA, dtype=str, keep_default_na=False, low_memory=False)
    out = pd.DataFrame(index=yr.index)
    out["source"] = "YRBSS"
    out["YEAR"] = 2023
    sex = pd.Series(np.nan, index=yr.index, dtype=object)
    sex[num(yr["Q2"]).eq(1)] = "female"
    sex[num(yr["Q2"]).eq(2)] = "male"
    out["sex"] = sex
    out["age_years"] = num(yr["Q1"]).map({1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 18})
    out["grade_9_12"] = category_code_string(num(yr["Q3"]).map({1: 9, 2: 10, 3: 11, 4: 12}))

    q87 = num(yr["Q87"])
    out["academic_achievement_3"] = np.select([q87.isin([1, 2]), q87.eq(3), q87.isin([4, 5])], [1.0, 2.0, 3.0], default=np.nan)
    out["body_image_5"] = num(yr["Q66"]).where(num(yr["Q66"]).between(1, 5), np.nan)
    height_m = num(yr["Q6"])
    weight_kg = num(yr["Q7"])
    bmi = weight_kg / (height_m**2)
    out["bmi"] = bmi.where(bmi.between(10, 60), np.nan)
    out["physical_activity_days"] = code_minus_one_days(yr["Q76"])
    out["physical_activity_5plus"] = binary_yes1_no2(yr["QN76"])
    out["breakfast_days"] = code_minus_one_days(yr["Q75"])
    out["trying_to_lose_weight"] = binary_from_bool(num(yr["Q67"]).eq(1), num(yr["Q67"]).between(1, 4))
    out["current_alcohol"] = binary_yes1_no2(yr["QN42"])
    out["binge_alcohol_any"] = binary_yes1_no2(yr["QN43"])
    out["lifetime_alcohol_experience"] = binary_from_bool(num(yr["Q41"]).between(2, 7), num(yr["Q41"]).between(1, 7))
    out["any_current_tobacco"] = binary_yes1_no2(yr["qntb4"])
    tobacco_parts = pd.concat([binary_yes1_no2(yr["Q31"]), binary_yes1_no2(yr["QN35"])], axis=1)
    out["lifetime_tobacco_any"] = tobacco_parts.max(axis=1, skipna=True)
    out["sexual_experience"] = binary_yes1_no2(yr["Q56"])
    out["sleep_hours"] = num(yr["Q85"]).map({1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10})
    out["sleep_8plus"] = binary_yes1_no2(yr["QN85"])
    out["digital_high_use"] = binary_yes1_no2(yr["QN80"])

    for target, yr_col in YRBSS_TARGETS.items():
        out[target] = binary_yes1_no2(yr[yr_col])
    out["depressed_mood_for_suicide_models"] = out["target_depressed_mood"]
    out["sample_weight"] = num(yr["weight"])
    out = out[out["grade_9_12"].isin(["9", "10", "11", "12"])].copy()
    return out


def feature_list(outcome_raw: str) -> list[str]:
    features = BASE_FEATURES.copy()
    if outcome_raw != "M_SAD":
        features.append("depressed_mood_for_suicide_models")
    return features


def weighted_metric_block(y: pd.Series, prob: np.ndarray, threshold: float, weight: pd.Series) -> dict[str, float]:
    valid = y.isin([0, 1]) & pd.Series(prob, index=y.index).notna() & weight.notna() & (weight > 0)
    yv = y[valid].astype(int)
    pv = prob[valid.to_numpy()]
    wv = weight[valid].astype(float)
    pred = (pv >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yv, pred, labels=[0, 1], sample_weight=wv).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    prevalence = np.average(yv, weights=wv) if len(yv) else np.nan
    return {
        "n_weighted_metric": int(len(yv)),
        "weighted_prevalence": float(prevalence),
        "weighted_auroc": float(roc_auc_score(yv, pv, sample_weight=wv)) if len(np.unique(yv)) == 2 else float("nan"),
        "weighted_auprc": float(average_precision_score(yv, pv, sample_weight=wv)),
        "weighted_brier": float(brier_score_loss(yv, pv, sample_weight=wv)),
        "weighted_sensitivity": float(recall_score(yv, pred, sample_weight=wv, zero_division=0)),
        "weighted_specificity": float(specificity),
        "weighted_precision": float(precision_score(yv, pred, sample_weight=wv, zero_division=0)),
        "weighted_f1": float(f1_score(yv, pred, sample_weight=wv, zero_division=0)),
        "weighted_f2": float(fbeta_score(yv, pred, beta=2, sample_weight=wv, zero_division=0)),
    }


def save_feature_missingness(kyrbs: pd.DataFrame, yrbss: pd.DataFrame) -> None:
    rows = []
    for feature in [*BASE_FEATURES, "depressed_mood_for_suicide_models"]:
        rows.append(
            {
                "feature": feature,
                "kyrbs_missing_rate": float(kyrbs[feature].isna().mean()),
                "kyrbs_missing_n": int(kyrbs[feature].isna().sum()),
                "yrbss_missing_rate": float(yrbss[feature].isna().mean()),
                "yrbss_missing_n": int(yrbss[feature].isna().sum()),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "tables" / "shared_feature_missingness.csv", index=False, encoding="utf-8-sig")


def optimize_external_one(
    kyrbs: pd.DataFrame,
    yrbss: pd.DataFrame,
    outcome_raw: str,
    model_name: str,
    trials: int,
    search_train_max: int,
    search_val_max: int,
    save_models: bool,
) -> dict[str, Any]:
    target_name = PRIMARY_OUTCOMES[outcome_raw]
    target_col = TARGET_COLS[outcome_raw]
    features = feature_list(outcome_raw)
    categorical = [f for f in features if f in CATEGORICAL_FEATURES]
    numeric = [f for f in features if f not in CATEGORICAL_FEATURES]

    model_data = kyrbs[kyrbs[target_col].isin([0, 1])].copy()
    train = model_data[model_data["YEAR"].between(2020, 2022)]
    val = model_data[model_data["YEAR"] == 2023]
    internal_test = model_data[model_data["YEAR"] == 2024]
    trainval = model_data[model_data["YEAR"].between(2020, 2023)]
    external = yrbss[yrbss[target_col].isin([0, 1])].copy()

    seed_material = f"external|{outcome_raw}|{model_name}|{RANDOM_SEED}".encode("utf-8")
    stable_seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
    rng = np.random.default_rng(stable_seed)
    x_search_train, y_search_train = stratified_limit(train[features], train[target_col].astype(int), search_train_max, stable_seed)
    x_search_val, y_search_val = stratified_limit(val[features], val[target_col].astype(int), search_val_max, stable_seed + 1)

    trial_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    t0 = time.time()
    for trial in range(1, trials + 1):
        model, params, scale = sample_model(model_name, rng, y_search_train)
        model, params = normalize_params(model, params)
        pipe = Pipeline(
            [
                ("preprocess", make_preprocessor(categorical, numeric, scale_numeric=scale)),
                ("model", model),
            ]
        )
        start = time.time()
        try:
            pipe.fit(x_search_train, y_search_train)
            prob = fit_predict_proba(pipe, x_search_val)
            row = {
                "outcome": target_name,
                "model": model_name,
                "trial": trial,
                "status": "ok",
                "search_val_auprc": float(average_precision_score(y_search_val, prob)),
                "search_val_auroc": float(roc_auc_score(y_search_val, prob)),
                "search_val_brier": float(brier_score_loss(y_search_val, prob)),
                "fit_seconds": round(time.time() - start, 3),
                "params": json.dumps(params, ensure_ascii=False, default=str),
            }
            if best is None or (row["search_val_auprc"], row["search_val_auroc"]) > (
                best["search_val_auprc"],
                best["search_val_auroc"],
            ):
                best = {**row, "params_obj": params, "scale": scale}
        except Exception as exc:
            row = {
                "outcome": target_name,
                "model": model_name,
                "trial": trial,
                "status": "failed",
                "error": repr(exc),
                "fit_seconds": round(time.time() - start, 3),
                "params": json.dumps(params, ensure_ascii=False, default=str),
            }
        trial_rows.append(row)
        if trial % 20 == 0:
            best_score = best["search_val_auprc"] if best else None
            print(f"[{target_name}/{model_name}] trial {trial}/{trials} best_auprc={best_score}", flush=True)

    if best is None:
        raise RuntimeError(f"All trials failed for {target_name}/{model_name}")

    best_train_model, _, scale = sample_model(model_name, np.random.default_rng(1), train[target_col].astype(int))
    best_train_model, _ = normalize_params(best_train_model, best["params_obj"])
    best_train_pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(categorical, numeric, scale_numeric=scale)),
            ("model", best_train_model),
        ]
    )
    best_train_pipe.fit(train[features], train[target_col].astype(int))
    val_prob = fit_predict_proba(best_train_pipe, val[features])
    thresholds = threshold_table(val[target_col].astype(int), val_prob)
    selected_rule = "youden" if target_name == "depressed_mood" else "f2"
    selected_threshold = thresholds[selected_rule]["threshold"]

    final_model, _, scale = sample_model(model_name, np.random.default_rng(1), trainval[target_col].astype(int))
    final_model, _ = normalize_params(final_model, best["params_obj"])
    final_pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(categorical, numeric, scale_numeric=scale)),
            ("model", final_model),
        ]
    )
    final_pipe.fit(trainval[features], trainval[target_col].astype(int))

    internal_prob = fit_predict_proba(final_pipe, internal_test[features])
    external_prob = fit_predict_proba(final_pipe, external[features])
    validation_metrics = metric_block(val[target_col].astype(int), val_prob, selected_threshold)
    internal_metrics = metric_block(internal_test[target_col].astype(int), internal_prob, selected_threshold)
    internal_metrics.update(calibration_slope_intercept(internal_test[target_col].astype(int), internal_prob))
    external_metrics = metric_block(external[target_col].astype(int), external_prob, selected_threshold)
    external_metrics.update(calibration_slope_intercept(external[target_col].astype(int), external_prob))
    external_weighted = weighted_metric_block(external[target_col].astype(int), external_prob, selected_threshold, external["sample_weight"])

    safe_name = f"{target_name}_{model_name}"
    pd.DataFrame(trial_rows).to_csv(OUT_DIR / "trials" / f"trials_{safe_name}.csv", index=False, encoding="utf-8-sig")
    feature_importance(final_pipe, model_name, top_n=80).to_csv(
        OUT_DIR / "feature_importance" / f"feature_importance_{safe_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            "source": "YRBSS",
            "y_true": external[target_col].astype(int).to_numpy(),
            "probability": external_prob,
            "sample_weight": external["sample_weight"].to_numpy(),
        }
    ).to_csv(OUT_DIR / "probabilities" / f"yrbss_probabilities_{safe_name}.csv", index=False, encoding="utf-8-sig")

    model_path = ""
    if save_models:
        model_path = str(OUT_DIR / "models" / f"model_{safe_name}.joblib")
        try:
            joblib.dump(final_pipe, model_path)
        except Exception as exc:
            model_path = f"MODEL_SAVE_FAILED: {exc!r}"

    return {
        "outcome": target_name,
        "outcome_raw": outcome_raw,
        "model": model_name,
        "trials_requested": trials,
        "trials_completed": int(sum(1 for r in trial_rows if r["status"] == "ok")),
        "features_n": len(features),
        "features": features,
        "categorical_features": categorical,
        "numeric_features": numeric,
        "selected_threshold_rule": selected_rule,
        "selected_threshold": float(selected_threshold),
        "threshold_youden": thresholds["youden"],
        "threshold_f2": thresholds["f2"],
        "best_validation_search": {k: v for k, v in best.items() if k not in {"params_obj", "scale"}},
        "best_params": best["params_obj"],
        "train_n": int(len(train)),
        "validation_n": int(len(val)),
        "internal_test_n": int(len(internal_test)),
        "external_test_n": int(len(external)),
        "train_prevalence": float(train[target_col].mean()),
        "validation_prevalence": float(val[target_col].mean()),
        "internal_test_prevalence": float(internal_test[target_col].mean()),
        "external_test_prevalence": float(external[target_col].mean()),
        "validation_metrics_at_selected_threshold": validation_metrics,
        "internal_2024_metrics_at_selected_threshold": internal_metrics,
        "external_yrbss_metrics_at_selected_threshold": external_metrics,
        "external_yrbss_weighted_metrics_at_selected_threshold": external_weighted,
        "model_path": model_path,
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def parse_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def write_summary(results: list[dict[str, Any]], kyrbs: pd.DataFrame, yrbss: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    for r in results:
        row = {
            "outcome": r["outcome"],
            "outcome_raw": r["outcome_raw"],
            "model": r["model"],
            "trials_completed": r["trials_completed"],
            "features_n": r["features_n"],
            "selected_threshold_rule": r["selected_threshold_rule"],
            "selected_threshold": r["selected_threshold"],
            "train_n": r["train_n"],
            "validation_n": r["validation_n"],
            "internal_test_n": r["internal_test_n"],
            "external_test_n": r["external_test_n"],
            "internal_prevalence": r["internal_test_prevalence"],
            "external_prevalence": r["external_test_prevalence"],
        }
        for prefix, metrics in [
            ("validation", r["validation_metrics_at_selected_threshold"]),
            ("internal_2024", r["internal_2024_metrics_at_selected_threshold"]),
            ("external_yrbss", r["external_yrbss_metrics_at_selected_threshold"]),
            ("external_yrbss_weighted", r["external_yrbss_weighted_metrics_at_selected_threshold"]),
        ]:
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["outcome", "external_yrbss_auprc", "external_yrbss_auroc"], ascending=[True, False, False])
    summary.to_csv(OUT_DIR / "external_validation_results_summary.csv", index=False, encoding="utf-8-sig")

    primary = summary.sort_values(["outcome", "validation_auprc", "validation_auroc"], ascending=[True, False, False])
    primary = primary.groupby("outcome", as_index=False).head(1)
    primary.to_csv(OUT_DIR / "primary_model_by_kyrbs_validation.csv", index=False, encoding="utf-8-sig")

    internal_best = summary.sort_values(["outcome", "internal_2024_auprc", "internal_2024_auroc"], ascending=[True, False, False])
    internal_best = internal_best.groupby("outcome", as_index=False).head(1)
    internal_best.to_csv(OUT_DIR / "secondary_model_by_kyrbs_2024_internal.csv", index=False, encoding="utf-8-sig")

    external_ranked = summary.sort_values(["outcome", "external_yrbss_auprc", "external_yrbss_auroc"], ascending=[True, False, False])
    external_ranked = external_ranked.groupby("outcome", as_index=False).head(1)
    external_ranked.to_csv(OUT_DIR / "best_external_model_by_outcome_descriptive_only.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# YRBSS 2023 external validation report",
        "",
        "## Design",
        "",
        "- KYRBS development population: grades 9-12 equivalent, KYRBS grade 3-6, years 2020-2024.",
        "- Tuning split: train 2020-2022, validation 2023.",
        "- Internal temporal check: KYRBS 2024.",
        "- External validation: CDC 2023 National YRBS high school dataset.",
        "- Thresholds are selected only on KYRBS 2023 validation data: Youden for depressed mood, F2 for suicidality outcomes.",
        "- External validation uses the shared-feature-only harmonized predictor set.",
        "",
        "## Cohort sizes",
        "",
        f"- KYRBS harmonized rows: {len(kyrbs):,}",
        f"- YRBSS harmonized rows: {len(yrbss):,}",
        "",
        "## Primary model selected by KYRBS validation AUPRC",
        "",
        "| Outcome | Model | Features | External AUROC | External AUPRC | Sensitivity | Specificity | Precision | Calibration intercept | Calibration slope |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in primary.iterrows():
        lines.append(
            "| {outcome} | {model} | {features_n:.0f} | {auroc:.3f} | {auprc:.3f} | {sens:.3f} | {spec:.3f} | {prec:.3f} | {cali:.3f} | {cals:.3f} |".format(
                outcome=row["outcome"],
                model=row["model"],
                features_n=row["features_n"],
                auroc=row["external_yrbss_auroc"],
                auprc=row["external_yrbss_auprc"],
                sens=row["external_yrbss_sensitivity"],
                spec=row["external_yrbss_specificity"],
                prec=row["external_yrbss_precision"],
                cali=row["external_yrbss_calibration_intercept"],
                cals=row["external_yrbss_calibration_slope"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation cautions",
            "",
            "- This is a transportability test, not a validation of the full KYRBS-only feature model.",
            "- Digital behavior is only approximately harmonized: KYRBS uses high smartphone time, while YRBSS uses frequent social-media use.",
            "- YRBSS does not provide direct matches for KYRBS GAD-7, loneliness, subjective economic status, or sleep satisfaction.",
            "- External calibration drift should be expected and should be followed by intercept/slope recalibration before any deployment claim.",
        ]
    )
    (OUT_DIR / "yrbss_external_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--outcomes", default="M_SAD,M_SUI_CON,M_SUI_PLN,M_SUI_ATT")
    parser.add_argument("--models", default="logistic,sgd_logistic,decision_tree,extra_trees,xgboost,lightgbm")
    parser.add_argument("--search-train-max", type=int, default=25000)
    parser.add_argument("--search-val-max", type=int, default=12000)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    kyrbs = kyrbs_shared_dataset()
    yrbss = yrbss_shared_dataset()
    kyrbs.to_csv(OUT_DIR / "tables" / "kyrbs_shared_feature_dataset.csv", index=False, encoding="utf-8-sig")
    yrbss.to_csv(OUT_DIR / "tables" / "yrbss_shared_feature_dataset.csv", index=False, encoding="utf-8-sig")
    save_feature_missingness(kyrbs, yrbss)

    config = {
        "trials": args.trials,
        "outcomes": parse_list(args.outcomes),
        "models": parse_list(args.models),
        "base_features": BASE_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "design": {
            "train": "KYRBS 2020-2022, grade 3-6 (grade 9-12 equivalent)",
            "validation": "KYRBS 2023, threshold selection and hyperparameter selection",
            "internal_temporal_test": "KYRBS 2024",
            "external_test": "YRBSS 2023 national high school",
        },
    }
    (OUT_DIR / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    results: list[dict[str, Any]] = []
    partial_path = OUT_DIR / "external_validation_results_partial.json"
    final_path = OUT_DIR / "external_validation_results.json"
    if args.resume:
        load_path = partial_path if partial_path.exists() else final_path
        if load_path.exists():
            results = json.loads(load_path.read_text(encoding="utf-8"))
            print(f"Resuming from {load_path} with {len(results)} completed rows", flush=True)

    completed = {(r["outcome_raw"], r["model"]) for r in results}
    failures: list[dict[str, Any]] = []
    for outcome in parse_list(args.outcomes):
        for model_name in parse_list(args.models):
            if (outcome, model_name) in completed:
                print(f"=== Skipping completed {outcome} / {model_name} ===", flush=True)
                continue
            print(f"=== Running external validation {outcome} / {model_name} ===", flush=True)
            try:
                result = optimize_external_one(
                    kyrbs=kyrbs,
                    yrbss=yrbss,
                    outcome_raw=outcome,
                    model_name=model_name,
                    trials=args.trials,
                    search_train_max=args.search_train_max,
                    search_val_max=args.search_val_max,
                    save_models=args.save_models,
                )
                results.append(result)
                partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            except Exception as exc:
                failure = {"outcome": outcome, "model": model_name, "error": repr(exc)}
                failures.append(failure)
                print(f"FAILED {failure}", flush=True)

    final_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if failures:
        (OUT_DIR / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        stale_failures = OUT_DIR / "failures.json"
        if stale_failures.exists():
            stale_failures.unlink()
    write_summary(results, kyrbs, yrbss)
    print(OUT_DIR / "external_validation_results_summary.csv")


if __name__ == "__main__":
    main()
