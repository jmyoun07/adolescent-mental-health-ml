from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".mplconfig"))

import joblib
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_protocol_ml import (
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


RAW_DIR = PROJECT / "data" / "raw_sav"
PROCESSED = PROJECT / "data" / "processed" / "kyrbs_2020_2025_protocol_analysis_dataset.csv"
DECISIONS = PROJECT / "reports" / "pubmed_lit_reduced_variable_decisions.csv"
OUT_DIR = PROJECT / "outputs" / "ml_lit_reduced_80trials"


OUTCOME_LABELS = {
    "M_SAD": "최근 12개월 동안 우울감 경험",
    "M_SUI_CON": "최근 12개월 동안 자살 생각",
    "M_SUI_PLN": "최근 12개월 동안 자살 계획",
    "M_SUI_ATT": "최근 12개월 동안 자살 시도",
}


RAW_SOURCE_MAP: dict[str, list[str]] = {
    "YEAR": ["YEAR"],
    "SEX": ["SEX"],
    "AGE": ["AGE"],
    "GRADE": ["GRADE"],
    "SCHOOL": ["SCHOOL"],
    "CTYPE": ["CTYPE"],
    "academic_achievement_3": ["E_S_RCRD"],
    "economic_status_3": ["E_SES"],
    "family_structure_3": ["A_FM", "E_FM_F_1", "E_FM_M_3"],
    "stress_3": ["M_STR"],
    "sleep_satisfaction_3": ["M_SLP_EN"],
    "avg_sleep_hours": ["M_SLP_HR", "M_SLP_MM", "M_WK_HR", "M_WK_MM", "M_SLP_HR_K", "M_SLP_MM_K", "M_WK_HR_K", "M_WK_MM_K"],
    "PR_HT": ["PR_HT"],
    "PR_BI": ["PR_BI"],
    "BMI": ["HT", "WT"],
    "smartphone_avg_daily_minutes": ["INT_SPWD_TM", "INT_SPWK_TM"],
    "sedentary_total_avg_minutes": ["PA_SWD_S", "PA_SWK_S", "PA_SWD_N", "PA_SWK_N"],
    "PA_TOT": ["PA_TOT"],
    "PA_MSC": ["PA_MSC"],
    "current_alcohol": ["AC_DAYS"],
    "heavy_drunk_any": ["AC_DRUNK"],
    "any_current_tobacco": ["TC_DAYS", "TC_EC_MN", "TC_HTP_MN"],
    "secondhand_smoke_exposure_any": ["TC_SND_H", "TC_SND_P"],
    "sexual_experience": ["S_SI"],
    "M_SAD_for_suicide_models": ["M_SAD"],
    "loneliness_3_or_M_LON": ["M_LON"],
    "gad7_total_or_high10": [f"M_GAD_{i}" for i in range(1, 8)],
    "drug_use_lifetime": ["DR_HAB_PUR"],
    "F_BR": ["F_BR"],
    "WC_MN": ["WC_MN"],
    "parent_college_or_higher_any": ["E_EDU_F", "E_EDU_M"],
    "school_injury_treatment_any": ["I_SCH_TRT"],
    "lifetime_alcohol_experience": ["AC_LT"],
    "lifetime_tobacco_any": ["TC_LT", "TC_EC_LT", "TC_HTP_LT"],
    "lifetime_cigarette_experience": ["TC_LT"],
    "violence_treatment_any": ["V_TRT"],
    "gad7_total": [f"M_GAD_{i}" for i in range(1, 8)],
    "gad7_high10": [f"M_GAD_{i}" for i in range(1, 8)],
    "loneliness_3": ["M_LON"],
    "loneliness_high": ["M_LON"],
    "asthma_lifetime": ["AS_DG_LT"],
    "allergic_rhinitis_lifetime": ["RH_DG_LT"],
    "atopic_dermatitis_lifetime": ["ECZ_DG_LT"],
    "atopic_disease_any": ["AS_DG_LT", "RH_DG_LT", "ECZ_DG_LT"],
    "atopic_disease_count": ["AS_DG_LT", "RH_DG_LT", "ECZ_DG_LT"],
    "substance_access_reported_any": ["AC_ACCESS", "TC_ACCESS"],
}


CATEGORICAL_FEATURES = {
    "SEX",
    "GRADE",
    "SCHOOL",
    "CTYPE",
    "family_structure_3",
}


@dataclass
class SplitData:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    x_trainval: pd.DataFrame
    y_trainval: pd.Series


def ensure_dirs() -> None:
    for sub in ["", "models", "feature_importance", "trials", "probabilities"]:
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def binary_yes2(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return pd.Series(np.where(s == 2, 1.0, np.where(s == 1, 0.0, np.nan)), index=s.index)


def current_from_conditional_days(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return pd.Series(np.where((s > 1) & (s < 9999), 1.0, np.where((s == 1) | (s == 9999), 0.0, np.nan)), index=s.index)


def clean_sentinel(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([7777, 8888, 9999], np.nan)


def recode_3(s: pd.Series) -> pd.Series:
    s = clean_sentinel(s)
    out = pd.Series(np.nan, index=s.index)
    out[s.isin([1, 2])] = 1.0
    out[s.eq(3)] = 2.0
    out[s.isin([4, 5])] = 3.0
    return out


def availability_for_sources(source_vars: list[str]) -> str:
    years = []
    for sav in sorted(RAW_DIR.glob("kyrbs202*.sav")):
        meta = pyreadstat.read_sav(str(sav), metadataonly=True)[1]
        available = set(meta.column_names)
        if all(v in available for v in source_vars):
            years.append(sav.stem[-4:])
    if not years:
        return "not available"
    if years == [str(y) for y in range(int(years[0]), int(years[-1]) + 1)]:
        return f"{years[0]}-{years[-1]}"
    return ",".join(years)


def read_raw_supplement() -> pd.DataFrame:
    needed = sorted({v for vars_ in RAW_SOURCE_MAP.values() for v in vars_})
    frames = []
    for sav in sorted(RAW_DIR.glob("kyrbs202*.sav")):
        year = int(sav.stem[-4:])
        meta = pyreadstat.read_sav(str(sav), metadataonly=True)[1]
        available = set(meta.column_names)
        usecols = [c for c in needed if c in available]
        frame, _ = pyreadstat.read_sav(str(sav), usecols=usecols, apply_value_formats=False)
        for col in needed:
            if col not in frame.columns:
                frame[col] = np.nan
        frame["YEAR_FROM_FILE"] = year
        frames.append(frame[["YEAR_FROM_FILE", *needed]])
    return pd.concat(frames, ignore_index=True)


def weighted_average(weekday: pd.Series, weekend: pd.Series) -> pd.Series:
    return (clean_sentinel(weekday) * 5 + clean_sentinel(weekend) * 2) / 7.0


def build_lit_dataset(force: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_dirs()
    dataset_path = PROJECT / "data" / "processed" / "kyrbs_2020_2025_lit_reduced_dataset.csv"
    metadata_path = PROJECT / "data" / "processed" / "kyrbs_2020_2025_lit_reduced_metadata.json"
    if dataset_path.exists() and metadata_path.exists() and not force:
        data = pd.read_csv(dataset_path, low_memory=False)
        with open(metadata_path, "r", encoding="utf-8") as f:
            return data, json.load(f)

    base = pd.read_csv(PROCESSED, low_memory=False)
    raw = read_raw_supplement()
    if len(base) != len(raw):
        raise RuntimeError(f"Row mismatch: processed={len(base)} raw_supplement={len(raw)}")

    df = pd.DataFrame(index=base.index)
    df["YEAR"] = pd.to_numeric(base["YEAR"], errors="coerce")
    for target in [
        "target_depressed_mood",
        "target_suicidal_ideation",
        "target_suicide_plan",
        "target_suicide_attempt_secondary",
    ]:
        df[target] = base[target]

    # Direct or compact recoded predictors from the protocol and decision table.
    for col in ["SEX", "AGE", "GRADE", "SCHOOL", "CTYPE", "PR_HT", "PR_BI", "PA_TOT", "PA_MSC", "F_BR", "WC_MN"]:
        df[col] = base[col] if col in base.columns else raw[col]
    df["academic_achievement_3"] = recode_3(base["E_S_RCRD"])
    df["economic_status_3"] = recode_3(base["E_SES"])

    father = clean_sentinel(base["E_FM_F_1"]).eq(1)
    mother = clean_sentinel(base["E_FM_M_3"]).eq(3)
    family = pd.Series(np.nan, index=base.index)
    family[father & mother] = 1.0
    family[father ^ mother] = 2.0
    family[(~father) & (~mother) & clean_sentinel(base["A_FM"]).notna()] = 3.0
    df["family_structure_3"] = family

    father_edu = clean_sentinel(base["E_EDU_F"])
    mother_edu = clean_sentinel(base["E_EDU_M"])
    parent_known = father_edu.notna() | mother_edu.notna()
    df["parent_college_or_higher_any"] = np.where(
        parent_known,
        (father_edu.isin([4, 5]) | mother_edu.isin([4, 5])).astype(float),
        np.nan,
    )

    df["stress_3"] = recode_3(base["M_STR"])
    df["sleep_satisfaction_3"] = recode_3(base["M_SLP_EN"])
    df["avg_sleep_hours"] = (base["sleep_weekday_hours"] * 5 + base["sleep_weekend_hours"] * 2) / 7.0
    df["BMI"] = base["BMI"]
    df["smartphone_avg_daily_minutes"] = base["smartphone_avg_daily_minutes"]
    df["sedentary_total_avg_minutes"] = weighted_average(base["PA_SWD_S"], base["PA_SWK_S"]) + weighted_average(
        base["PA_SWD_N"], base["PA_SWK_N"]
    )
    df["current_alcohol"] = base["current_alcohol"]
    df["heavy_drunk_any"] = base["heavy_drunk_any"]
    df["any_current_tobacco"] = base["any_current_tobacco"]
    df["secondhand_smoke_exposure_any"] = pd.concat(
        [base["secondhand_smoke_home"], base["secondhand_smoke_public"]], axis=1
    ).max(axis=1, skipna=True)
    df["sexual_experience"] = base["sexual_experience"]
    df["M_SAD_for_suicide_models"] = base["target_depressed_mood"]
    df["drug_use_lifetime"] = base["drug_use_lifetime"] if "drug_use_lifetime" in base.columns else np.nan

    df["school_injury_treatment_any"] = binary_yes2(raw["I_SCH_TRT"])
    df["lifetime_alcohol_experience"] = binary_yes2(base["AC_LT"])
    tobacco_parts = pd.DataFrame(
        {
            "cigarette": binary_yes2(base["TC_LT"]),
            "ecig": binary_yes2(base["TC_EC_LT"]),
            "htp": binary_yes2(base["TC_HTP_LT"]),
        }
    )
    df["lifetime_tobacco_any"] = tobacco_parts.max(axis=1, skipna=True)
    df["lifetime_cigarette_experience"] = tobacco_parts["cigarette"]
    df["violence_treatment_any"] = np.where(clean_sentinel(raw["V_TRT"]).notna(), (clean_sentinel(raw["V_TRT"]) > 1).astype(float), np.nan)

    gad_cols = [f"M_GAD_{i}" for i in range(1, 8)]
    gad = raw[gad_cols].apply(clean_sentinel)
    df["gad7_total"] = (gad - 1).sum(axis=1, min_count=7)
    df["gad7_high10"] = np.where(df["gad7_total"].notna(), (df["gad7_total"] >= 10).astype(float), np.nan)
    df["gad7_total_or_high10"] = df["gad7_total"]

    lon = clean_sentinel(raw["M_LON"])
    loneliness_3 = pd.Series(np.nan, index=raw.index)
    loneliness_3[lon.isin([1, 2])] = 1.0
    loneliness_3[lon.eq(3)] = 2.0
    loneliness_3[lon.isin([4, 5])] = 3.0
    df["loneliness_3"] = loneliness_3
    df["loneliness_high"] = np.where(lon.notna(), lon.isin([4, 5]).astype(float), np.nan)
    df["loneliness_3_or_M_LON"] = loneliness_3

    df["asthma_lifetime"] = binary_yes2(raw["AS_DG_LT"])
    df["allergic_rhinitis_lifetime"] = binary_yes2(raw["RH_DG_LT"])
    df["atopic_dermatitis_lifetime"] = binary_yes2(raw["ECZ_DG_LT"])
    atopic = df[["asthma_lifetime", "allergic_rhinitis_lifetime", "atopic_dermatitis_lifetime"]]
    df["atopic_disease_any"] = atopic.max(axis=1, skipna=True)
    df["atopic_disease_count"] = atopic.sum(axis=1, min_count=1)
    df["substance_access_reported_any"] = (
        clean_sentinel(base["AC_ACCESS"]).notna() | clean_sentinel(base["TC_ACCESS"]).notna()
    ).astype(float)

    decisions = pd.read_csv(DECISIONS, encoding="utf-8-sig")
    feature_names = [v for v in decisions["variable_name"].astype(str).tolist() if v in df.columns]
    missing_features = [v for v in decisions["variable_name"].astype(str).tolist() if v not in df.columns]

    source_rows = []
    for feature in decisions["variable_name"].astype(str):
        sources = RAW_SOURCE_MAP.get(feature, [feature])
        source_rows.append(
            {
                "variable_name": feature,
                "source_variables": ";".join(sources),
                "source_year_availability": availability_for_sources(sources),
                "in_lit_dataset": feature in df.columns,
                "decision_year_availability": decisions.loc[decisions["variable_name"].astype(str) == feature, "year_availability"].iloc[0],
                "missing_rate_in_lit_dataset": float(df[feature].isna().mean()) if feature in df.columns else np.nan,
            }
        )
    availability = pd.DataFrame(source_rows)
    availability.to_csv(PROJECT / "reports" / "lit_reduced_variable_year_availability_check.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "rows": int(len(df)),
        "decision_file": str(DECISIONS),
        "feature_names_in_decision_file": feature_names,
        "features_missing_from_lit_dataset": missing_features,
        "rules": {
            "outcome_positive_code": "KYRBS code 2 recoded as risk-positive for M_SAD/M_SUI_CON/M_SUI_PLN/M_SUI_ATT.",
            "splits": {
                "lit_compatible_2020_2025": "train=2020-2023, validation=2024, test=2025; excludes 2020-2024-only and 2022+-only predictors.",
                "lit_full_2020_2024": "train=2020-2022, validation=2023, test=2024; uses full decision-table predictors, allowing partial-missing variables such as drug_use_lifetime.",
            },
            "imputation": "Numeric median imputation with missing indicators; categorical missing is encoded as __MISSING__ before one-hot encoding.",
            "threshold_selection": "Depressed mood uses Youden threshold on validation; suicidality outcomes use F2 threshold to prioritize sensitivity.",
            "optimization": "80 random-search trials per model/outcome/feature set, selected by validation AUPRC then AUROC.",
        },
    }
    df.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return df, metadata


def load_decision_features(feature_set: str, outcome_raw: str) -> list[str]:
    decisions = pd.read_csv(DECISIONS, encoding="utf-8-sig")
    rows = decisions.copy()
    if outcome_raw == "M_SAD":
        rows = rows[rows["variable_name"] != "M_SAD_for_suicide_models"]
    if feature_set == "lit_compatible_2020_2025":
        excluded_scopes = rows["use_scope"].astype(str).str.contains("2020-2024|2022-2025|single-year", regex=True)
        excluded_names = rows["variable_name"].isin(
            [
                "loneliness_3_or_M_LON",
                "gad7_total_or_high10",
                "drug_use_lifetime",
                "violence_treatment_any",
                "gad7_total",
                "gad7_high10",
                "loneliness_3",
                "loneliness_high",
                "asthma_lifetime",
                "allergic_rhinitis_lifetime",
                "atopic_dermatitis_lifetime",
                "atopic_disease_any",
                "atopic_disease_count",
            ]
        )
        rows = rows[~excluded_scopes & ~excluded_names]
    elif feature_set == "lit_full_2020_2024":
        pass
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    return rows["variable_name"].astype(str).tolist()


def feature_columns(df: pd.DataFrame, feature_set: str, outcome_raw: str) -> tuple[list[str], list[str], list[str]]:
    requested = load_decision_features(feature_set, outcome_raw)
    available = [f for f in requested if f in df.columns]
    categorical = [f for f in available if f in CATEGORICAL_FEATURES]
    numeric = [f for f in available if f not in CATEGORICAL_FEATURES]
    return available, categorical, numeric


def make_preprocessor(categorical: list[str], numeric: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("to_string", FunctionTransformer(lambda x: x.astype(str), feature_names_out="one-to-one")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("num", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(("cat", categorical_pipe, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_split(df: pd.DataFrame, target_col: str, feature_set: str, features: list[str]) -> SplitData:
    keep_cols = list(dict.fromkeys(features + ["YEAR", target_col]))
    data = df.loc[:, keep_cols].copy()
    data = data[data[target_col].isin([0, 1])]
    if feature_set == "lit_compatible_2020_2025":
        train = data[data["YEAR"].between(2020, 2023)]
        val = data[data["YEAR"] == 2024]
        test = data[data["YEAR"] == 2025]
        trainval = data[data["YEAR"].between(2020, 2024)]
    elif feature_set == "lit_full_2020_2024":
        data = data[data["YEAR"].between(2020, 2024)]
        train = data[data["YEAR"].between(2020, 2022)]
        val = data[data["YEAR"] == 2023]
        test = data[data["YEAR"] == 2024]
        trainval = data[data["YEAR"].between(2020, 2023)]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    return SplitData(
        x_train=train[features],
        y_train=train[target_col].astype(int),
        x_val=val[features],
        y_val=val[target_col].astype(int),
        x_test=test[features],
        y_test=test[target_col].astype(int),
        x_trainval=trainval[features],
        y_trainval=trainval[target_col].astype(int),
    )


def safe_auc(y: pd.Series, prob: np.ndarray) -> float:
    return float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan")


def stratified_limit(x: pd.DataFrame, y: pd.Series, max_n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    if max_n <= 0 or len(y) <= max_n:
        return x, y
    rng = np.random.default_rng(seed)
    y_array = y.to_numpy()
    keep_idx = []
    for cls in np.unique(y_array):
        idx = np.flatnonzero(y_array == cls)
        n_cls = max(1, int(round(max_n * len(idx) / len(y_array))))
        n_cls = min(n_cls, len(idx))
        keep_idx.extend(rng.choice(idx, size=n_cls, replace=False).tolist())
    keep_idx = np.array(sorted(keep_idx))
    return x.iloc[keep_idx], y.iloc[keep_idx]


def fit_predict_proba(pipe: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        return pipe.predict_proba(x)[:, 1]
    scores = pipe.decision_function(x)
    return 1.0 / (1.0 + np.exp(-scores))


def normalize_params(model: Any, params: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    clean: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            value = value.item()
        if value == "None":
            value = None
        clean[key] = value
    model.set_params(**clean)
    return model, clean


def optimize_one(
    df: pd.DataFrame,
    outcome_raw: str,
    feature_set: str,
    model_name: str,
    trials: int,
    save_models: bool,
    search_train_max: int,
    search_val_max: int,
) -> dict[str, Any]:
    target_name = PRIMARY_OUTCOMES[outcome_raw]
    target_col = f"target_{target_name}"
    features, categorical, numeric = feature_columns(df, feature_set, outcome_raw)
    split = make_split(df, target_col, feature_set, features)
    seed_material = f"{outcome_raw}|{feature_set}|{model_name}|{RANDOM_SEED}".encode("utf-8")
    stable_seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
    rng = np.random.default_rng(stable_seed)
    x_search_train, y_search_train = stratified_limit(split.x_train, split.y_train, search_train_max, stable_seed)
    x_search_val, y_search_val = stratified_limit(split.x_val, split.y_val, search_val_max, stable_seed + 1)

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
                "feature_set": feature_set,
                "model": model_name,
                "trial": trial,
                "status": "ok",
                "search_val_auprc": float(average_precision_score(y_search_val, prob)),
                "search_val_auroc": safe_auc(y_search_val, prob),
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
                "feature_set": feature_set,
                "model": model_name,
                "trial": trial,
                "status": "failed",
                "error": repr(exc),
                "fit_seconds": round(time.time() - start, 3),
                "params": json.dumps(params, ensure_ascii=False, default=str),
            }
        trial_rows.append(row)
        if trial % 20 == 0:
            print(
                f"[{target_name}/{feature_set}/{model_name}] trial {trial}/{trials} "
                f"best_auprc={best['search_val_auprc'] if best else None}",
                flush=True,
            )

    if best is None:
        pd.DataFrame(trial_rows).to_csv(
            OUT_DIR / "trials" / f"trials_failed_{target_name}_{feature_set}_{model_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise RuntimeError(f"All trials failed for {target_name}/{feature_set}/{model_name}")

    best_train_model, _, scale = sample_model(model_name, np.random.default_rng(1), split.y_train)
    best_train_model, _ = normalize_params(best_train_model, best["params_obj"])
    best_train_pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(categorical, numeric, scale_numeric=scale)),
            ("model", best_train_model),
        ]
    )
    best_train_pipe.fit(split.x_train, split.y_train)
    full_val_prob = fit_predict_proba(best_train_pipe, split.x_val)
    thresholds = threshold_table(split.y_val, full_val_prob)
    selected_rule = "youden" if target_name == "depressed_mood" else "f2"
    selected_threshold = thresholds[selected_rule]["threshold"]

    final_model, _, scale = sample_model(model_name, np.random.default_rng(1), split.y_trainval)
    final_model, _ = normalize_params(final_model, best["params_obj"])
    final_pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(categorical, numeric, scale_numeric=scale)),
            ("model", final_model),
        ]
    )
    final_pipe.fit(split.x_trainval, split.y_trainval)
    test_prob = fit_predict_proba(final_pipe, split.x_test)
    val_metrics = metric_block(split.y_val, full_val_prob, selected_threshold)
    test_metrics = metric_block(split.y_test, test_prob, selected_threshold)
    test_metrics.update(calibration_slope_intercept(split.y_test, test_prob))

    safe_name = f"{target_name}_{feature_set}_{model_name}"
    pd.DataFrame(trial_rows).to_csv(OUT_DIR / "trials" / f"trials_{safe_name}.csv", index=False, encoding="utf-8-sig")
    feature_importance(final_pipe, model_name, top_n=80).to_csv(
        OUT_DIR / "feature_importance" / f"feature_importance_{safe_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            "YEAR": split.x_test.index.map(lambda _: np.nan),
            "y_true": split.y_test.to_numpy(),
            "probability": test_prob,
        }
    ).to_csv(OUT_DIR / "probabilities" / f"test_probabilities_{safe_name}.csv", index=False, encoding="utf-8-sig")

    model_path = ""
    if save_models:
        model_path = str(OUT_DIR / "models" / f"model_{safe_name}.joblib")
        joblib.dump(final_pipe, model_path)

    return {
        "outcome": target_name,
        "outcome_raw": outcome_raw,
        "outcome_label": OUTCOME_LABELS[outcome_raw],
        "feature_set": feature_set,
        "model": model_name,
        "trials_requested": trials,
        "trials_completed": int(sum(1 for r in trial_rows if r["status"] == "ok")),
        "features_n": len(features),
        "features": features,
        "categorical_features": categorical,
        "numeric_features": numeric,
        "search_train_n": int(len(y_search_train)),
        "search_validation_n": int(len(y_search_val)),
        "train_n": int(len(split.y_train)),
        "validation_n": int(len(split.y_val)),
        "test_n": int(len(split.y_test)),
        "train_prevalence": float(split.y_train.mean()),
        "validation_prevalence": float(split.y_val.mean()),
        "test_prevalence": float(split.y_test.mean()),
        "selected_threshold_rule": selected_rule,
        "selected_threshold": float(selected_threshold),
        "threshold_youden": thresholds["youden"],
        "threshold_f2": thresholds["f2"],
        "best_validation_search": {k: v for k, v in best.items() if k not in {"params_obj", "scale"}},
        "best_params": best["params_obj"],
        "validation_metrics_at_selected_threshold": val_metrics,
        "test_metrics_at_selected_threshold": test_metrics,
        "model_path": model_path,
        "elapsed_seconds": round(time.time() - t0, 3),
    }


def parse_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--outcomes", default="M_SAD,M_SUI_CON,M_SUI_PLN,M_SUI_ATT")
    parser.add_argument("--feature-sets", default="lit_compatible_2020_2025,lit_full_2020_2024")
    parser.add_argument("--models", default="logistic,sgd_logistic,decision_tree,extra_trees,xgboost,lightgbm")
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--search-train-max", type=int, default=25000)
    parser.add_argument("--search-val-max", type=int, default=12000)
    args = parser.parse_args()

    ensure_dirs()
    df, metadata = build_lit_dataset(force=args.force_dataset)
    config = {
        "trials": args.trials,
        "outcomes": parse_list(args.outcomes),
        "feature_sets": parse_list(args.feature_sets),
        "models": parse_list(args.models),
        "metadata": metadata,
    }
    with open(OUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)

    results: list[dict[str, Any]] = []
    if args.resume:
        partial_path = OUT_DIR / "model_results_partial.json"
        final_path = OUT_DIR / "model_results.json"
        load_path = partial_path if partial_path.exists() else final_path
        if load_path.exists():
            with open(load_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Resuming from {load_path} with {len(results)} completed result rows", flush=True)
    completed = {(r["outcome_raw"], r["feature_set"], r["model"]) for r in results}
    failures: list[dict[str, Any]] = []
    for outcome in parse_list(args.outcomes):
        for feature_set in parse_list(args.feature_sets):
            for model_name in parse_list(args.models):
                if (outcome, feature_set, model_name) in completed:
                    print(f"=== Skipping completed {outcome} / {feature_set} / {model_name} ===", flush=True)
                    continue
                print(f"=== Running {outcome} / {feature_set} / {model_name} ===", flush=True)
                try:
                    result = optimize_one(
                        df=df,
                        outcome_raw=outcome,
                        feature_set=feature_set,
                        model_name=model_name,
                        trials=args.trials,
                        save_models=args.save_models,
                        search_train_max=args.search_train_max,
                        search_val_max=args.search_val_max,
                    )
                    results.append(result)
                    with open(OUT_DIR / "model_results_partial.json", "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
                except Exception as exc:
                    failure = {
                        "outcome": outcome,
                        "feature_set": feature_set,
                        "model": model_name,
                        "error": repr(exc),
                    }
                    failures.append(failure)
                    print(f"FAILED {failure}", flush=True)

    with open(OUT_DIR / "model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    if failures:
        with open(OUT_DIR / "failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2, default=str)
    else:
        stale_failures = OUT_DIR / "failures.json"
        if stale_failures.exists():
            stale_failures.unlink()

    rows: list[dict[str, Any]] = []
    for r in results:
        row = {
            "outcome": r["outcome"],
            "feature_set": r["feature_set"],
            "model": r["model"],
            "trials_completed": r["trials_completed"],
            "features_n": r["features_n"],
            "selected_threshold_rule": r["selected_threshold_rule"],
            "selected_threshold": r["selected_threshold"],
            "train_n": r["train_n"],
            "validation_n": r["validation_n"],
            "test_n": r["test_n"],
            "test_prevalence": r["test_prevalence"],
            "search_val_auprc": r["best_validation_search"].get("search_val_auprc"),
            "search_val_auroc": r["best_validation_search"].get("search_val_auroc"),
        }
        for prefix, metrics in [
            ("val", r["validation_metrics_at_selected_threshold"]),
            ("test", r["test_metrics_at_selected_threshold"]),
        ]:
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    if rows:
        summary = pd.DataFrame(rows).sort_values(["outcome", "feature_set", "test_auprc"], ascending=[True, True, False])
    else:
        summary = pd.DataFrame()
    summary.to_csv(OUT_DIR / "model_results_summary.csv", index=False, encoding="utf-8-sig")
    print(OUT_DIR / "model_results_summary.csv")


if __name__ == "__main__":
    main()
