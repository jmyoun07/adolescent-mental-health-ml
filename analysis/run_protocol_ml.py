from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyreadstat
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


RANDOM_SEED = 20260701

PRIMARY_OUTCOMES = {
    "M_SAD": "depressed_mood",
    "M_SUI_CON": "suicidal_ideation",
    "M_SUI_PLN": "suicide_plan",
    "M_SUI_ATT": "suicide_attempt_secondary",
}

OUTCOME_LABELS = {
    "M_SAD": "최근 12개월 동안 우울감 경험",
    "M_SUI_CON": "최근 12개월 동안 자살 생각",
    "M_SUI_PLN": "최근 12개월 동안 자살 계획",
    "M_SUI_ATT": "최근 12개월 동안 자살 시도",
}

RAW_COLUMNS = [
    "YEAR",
    "CITY",
    "CTYPE",
    "SCHOOL",
    "STYPE",
    "GRADE",
    "AGE",
    "AGE_M",
    "SEX",
    "W",
    "STRATA",
    "CLUSTER",
    "FPC",
    "M_STR",
    "M_SLP_EN",
    "M_SLP_HR",
    "M_SLP_MM",
    "M_WK_HR",
    "M_WK_MM",
    "M_SLP_HR_K",
    "M_SLP_MM_K",
    "M_WK_HR_K",
    "M_WK_MM_K",
    "M_SAD",
    "M_SUI_CON",
    "M_SUI_PLN",
    "M_SUI_ATT",
    "INT_SPWD",
    "INT_SPWD_TM",
    "INT_SPWK",
    "INT_SPWK_TM",
    "PA_SWD_S",
    "PA_SWD_N",
    "PA_SWK_S",
    "PA_SWK_N",
    "PA_TOT",
    "PA_MSC",
    "F_BR",
    "F_FRUIT",
    "F_FASTFOOD",
    "F_WAT",
    "PR_HT",
    "PR_BI",
    "HT",
    "WT",
    "WC_MN",
    "AC_LT",
    "AC_DAYS",
    "AC_AMNT",
    "AC_DRUNK",
    "AC_ACCESS",
    "TC_LT",
    "TC_DAYS",
    "TC_AMNT",
    "TC_EC_LT",
    "TC_EC_MN",
    "TC_HTP_LT",
    "TC_HTP_MN",
    "TC_HTP_AMNT",
    "TC_ACCESS",
    "TC_QT_YR",
    "TC_SND_H",
    "TC_SND_P",
    "E_S_RCRD",
    "E_SES",
    "E_RES",
    "A_FM",
    "E_FM_F_1",
    "E_FM_SF_2",
    "E_FM_M_3",
    "E_FM_SM_4",
    "E_FM_GF_5",
    "E_FM_GM_6",
    "E_FM_OBS_7",
    "E_FM_YBS_8",
    "E_FM_NO_9",
    "E_EDU_F",
    "E_EDU_M",
    "E_KRN_F",
    "E_KRN_M",
    "I_SB_FR",
    "I_SB_BK",
    "I_SB_EX",
    "S_SI",
    "S_CONT",
    "O_BR_S",
]

OPTIONAL_RAW_COLUMNS = [
    "DR_HAB_PUR",
    "F_CAFF_A",
    "F_SWD_A",
    "F_ZERO",
    "F_VEG_A",
    "F_MD",
    "F_NE",
    "PA_VIG_D",
]

CATEGORICAL_BASE = [
    "CITY",
    "CTYPE",
    "SCHOOL",
    "STYPE",
    "SEX",
    "GRADE",
    "E_RES",
    "A_FM",
    "E_FM_F_1",
    "E_FM_SF_2",
    "E_FM_M_3",
    "E_FM_SM_4",
    "E_FM_GF_5",
    "E_FM_GM_6",
    "E_FM_OBS_7",
    "E_FM_YBS_8",
    "E_FM_NO_9",
    "E_EDU_F",
    "E_EDU_M",
    "E_KRN_F",
    "E_KRN_M",
]

NUMERIC_STANDARD = [
    "AGE",
    "AGE_M",
    "M_STR",
    "M_SLP_EN",
    "sleep_weekday_hours",
    "sleep_weekend_hours",
    "sleep_weekday_midpoint",
    "sleep_weekend_midpoint",
    "PA_TOT",
    "PA_MSC",
    "PA_SWD_S",
    "PA_SWD_N",
    "PA_SWK_S",
    "PA_SWK_N",
    "F_BR",
    "F_FRUIT",
    "F_FASTFOOD",
    "F_WAT",
    "PR_HT",
    "PR_BI",
    "BMI",
    "WC_MN",
    "E_S_RCRD",
    "E_SES",
    "current_alcohol",
    "alcohol_days_level",
    "heavy_drunk_any",
    "alcohol_access",
    "current_cigarette",
    "cigarette_days_level",
    "current_ecig",
    "current_htp",
    "any_current_tobacco",
    "secondhand_smoke_home",
    "secondhand_smoke_public",
    "front_seatbelt",
    "back_seatbelt",
    "bus_seatbelt",
    "sexual_experience",
    "contraception_use",
    "lunch_toothbrushing",
]

NUMERIC_DIGITAL = [
    "smartphone_weekday_minutes",
    "smartphone_weekend_minutes",
    "smartphone_avg_daily_minutes",
    "smartphone_weekend_minus_weekday",
    "smartphone_weekday_used",
    "smartphone_weekend_used",
    "high_smartphone_4h",
    "high_smartphone_6h",
]


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


def project_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ensure_dirs(project: str) -> dict[str, str]:
    dirs = {
        "reports": os.path.join(project, "reports"),
        "processed": os.path.join(project, "data", "processed"),
        "outputs": os.path.join(project, "outputs", "ml_protocol_80trials"),
        "models": os.path.join(project, "outputs", "ml_protocol_80trials", "models"),
        "figures": os.path.join(project, "outputs", "ml_protocol_80trials", "figures"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def clip_or_nan(s: pd.Series, low: float, high: float) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s >= low) & (s <= high), np.nan)


def binary_yes2(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return np.where(s == 2, 1.0, np.where(s == 1, 0.0, np.nan))


def current_from_conditional_days(s: pd.Series) -> pd.Series:
    # KYRBS conditional frequency variables use 9999 for skipped/not applicable.
    # Code 1 is treated as no recent use; codes above 1 and below 9999 indicate use.
    s = pd.to_numeric(s, errors="coerce")
    return np.where((s > 1) & (s < 9999), 1.0, np.where((s == 1) | (s == 9999), 0.0, np.nan))


def level_from_conditional(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return np.where(s == 9999, np.nan, s)


def minutes_from_time(hour: pd.Series, minute: pd.Series) -> pd.Series:
    h = pd.to_numeric(hour, errors="coerce")
    m = pd.to_numeric(minute, errors="coerce")
    valid = (h >= 0) & (h <= 23) & (m >= 0) & (m <= 59)
    return (h * 60 + m).where(valid, np.nan)


def sleep_duration_hours(bed: pd.Series, wake: pd.Series) -> pd.Series:
    duration = wake - bed
    duration = duration.where(duration > 0, duration + 24 * 60)
    hours = duration / 60.0
    return hours.where((hours >= 2) & (hours <= 16), np.nan)


def sleep_midpoint_hour(bed: pd.Series, duration_hours: pd.Series) -> pd.Series:
    midpoint = (bed + duration_hours * 30.0) % (24 * 60)
    return midpoint / 60.0


def build_analysis_dataset(project: str, force: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    dirs = ensure_dirs(project)
    dataset_path = os.path.join(dirs["processed"], "kyrbs_2020_2025_protocol_analysis_dataset.csv")
    pickle_path = os.path.join(dirs["processed"], "kyrbs_2020_2025_protocol_analysis_dataset.pkl")
    metadata_path = os.path.join(dirs["processed"], "kyrbs_2020_2025_protocol_analysis_metadata.json")
    if os.path.exists(metadata_path) and not force:
        if os.path.exists(pickle_path):
            df = pd.read_pickle(pickle_path)
        elif os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path, low_memory=False)
            df.to_pickle(pickle_path)
        else:
            df = None
        if df is not None:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return df, metadata
    if os.path.exists(dataset_path) and os.path.exists(metadata_path) and not force:
        df = pd.read_csv(dataset_path, low_memory=False)
        df.to_pickle(pickle_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return df, metadata

    raw_dir = os.path.join(project, "data", "raw_sav")
    frames: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {"years": {}, "rules": {}}
    for filename in sorted(os.listdir(raw_dir)):
        if not filename.lower().endswith(".sav"):
            continue
        path = os.path.join(raw_dir, filename)
        year = int("".join(ch for ch in filename if ch.isdigit())[:4])
        _, meta = pyreadstat.read_sav(path, metadataonly=True, apply_value_formats=False)
        available = set(meta.column_names)
        usecols = [c for c in RAW_COLUMNS + OPTIONAL_RAW_COLUMNS if c in available]
        df_year, meta_year = pyreadstat.read_sav(path, usecols=usecols, apply_value_formats=False)
        if "YEAR" not in df_year.columns:
            df_year["YEAR"] = year
        frames.append(df_year)
        metadata["years"][str(year)] = {
            "file": filename,
            "rows": int(len(df_year)),
            "columns_read": len(usecols),
            "column_labels": {
                k: v
                for k, v in (meta_year.column_names_to_labels or {}).items()
                if k in usecols
            },
        }

    df = pd.concat(frames, ignore_index=True)
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")

    # Outcomes. In these KYRBS binary items, code 2 is the affirmative risk response.
    for raw, name in PRIMARY_OUTCOMES.items():
        if raw in df.columns:
            df[f"target_{name}"] = binary_yes2(df[raw])

    # Digital phenotype.
    for col in ["INT_SPWD_TM", "INT_SPWK_TM"]:
        if col in df.columns:
            df[col] = clip_or_nan(df[col], 0, 1440)
    df["smartphone_weekday_minutes"] = df.get("INT_SPWD_TM", pd.Series(np.nan, index=df.index)).fillna(0)
    df["smartphone_weekend_minutes"] = df.get("INT_SPWK_TM", pd.Series(np.nan, index=df.index)).fillna(0)
    df["smartphone_avg_daily_minutes"] = (
        df["smartphone_weekday_minutes"] * 5 + df["smartphone_weekend_minutes"] * 2
    ) / 7.0
    df["smartphone_weekend_minus_weekday"] = (
        df["smartphone_weekend_minutes"] - df["smartphone_weekday_minutes"]
    )
    df["smartphone_weekday_used"] = (df["smartphone_weekday_minutes"] > 0).astype(float)
    df["smartphone_weekend_used"] = (df["smartphone_weekend_minutes"] > 0).astype(float)
    df["high_smartphone_4h"] = (df["smartphone_avg_daily_minutes"] >= 240).astype(float)
    df["high_smartphone_6h"] = (df["smartphone_avg_daily_minutes"] >= 360).astype(float)

    # Sleep duration and timing.
    weekday_bed = minutes_from_time(df.get("M_SLP_HR"), df.get("M_SLP_MM"))
    weekday_wake = minutes_from_time(df.get("M_WK_HR"), df.get("M_WK_MM"))
    weekend_bed = minutes_from_time(df.get("M_SLP_HR_K"), df.get("M_SLP_MM_K"))
    weekend_wake = minutes_from_time(df.get("M_WK_HR_K"), df.get("M_WK_MM_K"))
    df["sleep_weekday_hours"] = sleep_duration_hours(weekday_bed, weekday_wake)
    df["sleep_weekend_hours"] = sleep_duration_hours(weekend_bed, weekend_wake)
    df["sleep_weekday_midpoint"] = sleep_midpoint_hour(weekday_bed, df["sleep_weekday_hours"])
    df["sleep_weekend_midpoint"] = sleep_midpoint_hour(weekend_bed, df["sleep_weekend_hours"])

    # Body composition.
    ht = clip_or_nan(df.get("HT"), 100, 220)
    wt = clip_or_nan(df.get("WT"), 20, 200)
    bmi = wt / ((ht / 100.0) ** 2)
    df["BMI"] = bmi.where((bmi >= 10) & (bmi <= 60), np.nan)

    # Substance/risk behavior recoding.
    df["current_alcohol"] = current_from_conditional_days(df.get("AC_DAYS"))
    df["alcohol_days_level"] = level_from_conditional(df.get("AC_DAYS"))
    df["heavy_drunk_any"] = current_from_conditional_days(df.get("AC_DRUNK"))
    df["alcohol_access"] = level_from_conditional(df.get("AC_ACCESS"))
    df["current_cigarette"] = current_from_conditional_days(df.get("TC_DAYS"))
    df["cigarette_days_level"] = level_from_conditional(df.get("TC_DAYS"))
    df["current_ecig"] = current_from_conditional_days(df.get("TC_EC_MN"))
    df["current_htp"] = current_from_conditional_days(df.get("TC_HTP_MN"))
    df["any_current_tobacco"] = (
        (df["current_cigarette"].fillna(0) + df["current_ecig"].fillna(0) + df["current_htp"].fillna(0)) > 0
    ).astype(float)
    df["secondhand_smoke_home"] = current_from_conditional_days(df.get("TC_SND_H"))
    df["secondhand_smoke_public"] = current_from_conditional_days(df.get("TC_SND_P"))
    df["front_seatbelt"] = level_from_conditional(df.get("I_SB_FR"))
    df["back_seatbelt"] = level_from_conditional(df.get("I_SB_BK"))
    df["bus_seatbelt"] = level_from_conditional(df.get("I_SB_EX"))
    df["sexual_experience"] = binary_yes2(df.get("S_SI"))
    df["contraception_use"] = binary_yes2(df.get("S_CONT"))
    df["lunch_toothbrushing"] = level_from_conditional(df.get("O_BR_S"))

    if "DR_HAB_PUR" in df.columns:
        df["drug_use_lifetime"] = binary_yes2(df["DR_HAB_PUR"])

    metadata["rules"] = {
        "split": "Train=2020-2023, validation=2024, holdout test=2025",
        "outcome_positive_code": "2 is recoded as positive for M_SAD, M_SUI_CON, M_SUI_PLN, M_SUI_ATT",
        "conditional_9999": "9999 is treated as not-applicable/skip. Current-use binary features recode 9999 to 0; conditional intensity variables set 9999 to missing.",
        "smartphone_missing_time": "Missing smartphone minutes are treated as 0 minutes, consistent with non-use skip patterns; continuous values are capped to 0-1440 minutes.",
        "sleep_duration": "Bed/wake times are converted to hours; durations outside 2-16 hours are set missing.",
        "bmi": "Height outside 100-220 cm, weight outside 20-200 kg, and BMI outside 10-60 are set missing.",
    }
    metadata["dataset"] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target_prevalence": {
            name: float(df[f"target_{name}"].mean(skipna=True))
            for name in PRIMARY_OUTCOMES.values()
            if f"target_{name}" in df.columns
        },
    }
    df = df.copy()
    df.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    df.to_pickle(pickle_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return df, metadata


def feature_columns(feature_set: str, available: set[str]) -> tuple[list[str], list[str]]:
    categorical = [c for c in CATEGORICAL_BASE if c in available]
    standard_numeric = [c for c in NUMERIC_STANDARD if c in available]
    digital_numeric = [c for c in NUMERIC_DIGITAL if c in available]
    if feature_set == "digital_only":
        return [], digital_numeric
    if feature_set == "standard":
        return categorical, standard_numeric
    if feature_set == "full":
        return categorical, standard_numeric + digital_numeric
    raise ValueError(f"Unknown feature_set: {feature_set}")


def make_preprocessor(categorical: list[str], numeric: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("num", numeric_pipe, numeric))
    if categorical:
        transformers.append(("cat", categorical_pipe, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_split(df: pd.DataFrame, target_col: str, categorical: list[str], numeric: list[str]) -> SplitData:
    keep_cols = categorical + numeric + ["YEAR", target_col]
    data = df.loc[:, keep_cols].copy()
    data = data[data[target_col].isin([0, 1])]
    train = data[data["YEAR"].between(2020, 2023)]
    val = data[data["YEAR"] == 2024]
    test = data[data["YEAR"] == 2025]
    trainval = data[data["YEAR"].between(2020, 2024)]
    x_cols = categorical + numeric
    return SplitData(
        x_train=train[x_cols],
        y_train=train[target_col].astype(int),
        x_val=val[x_cols],
        y_val=val[target_col].astype(int),
        x_test=test[x_cols],
        y_test=test[target_col].astype(int),
        x_trainval=trainval[x_cols],
        y_trainval=trainval[target_col].astype(int),
    )


def loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(10 ** rng.uniform(math.log10(low), math.log10(high)))


def randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def sample_choice(rng: np.random.Generator, values: list[Any]) -> Any:
    return values[int(rng.integers(0, len(values)))]


def sample_model(model_name: str, rng: np.random.Generator, y_train: pd.Series) -> tuple[Any, dict[str, Any], bool]:
    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
    ratio = neg / pos
    if model_name == "logistic":
        params: dict[str, Any] = {
            "C": loguniform(rng, 0.005, 50),
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 700,
            "n_jobs": -1,
            "random_state": RANDOM_SEED,
            "class_weight": rng.choice([None, "balanced"]),
        }
        model = LogisticRegression(**params)
        return model, params, True
    if model_name == "sgd_logistic":
        params = {
            "loss": "log_loss",
            "penalty": rng.choice(["l2", "l1", "elasticnet"]),
            "alpha": loguniform(rng, 1e-6, 1e-2),
            "l1_ratio": float(rng.uniform(0.05, 0.95)),
            "max_iter": randint(rng, 800, 2000),
            "tol": loguniform(rng, 1e-5, 1e-3),
            "class_weight": rng.choice([None, "balanced"]),
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
        model = SGDClassifier(**params)
        return model, params, True
    if model_name == "random_forest":
        params = {
            "n_estimators": randint(rng, 80, 240),
            "max_depth": rng.choice([None, 4, 6, 8, 10, 12, 16, 20]),
            "min_samples_split": randint(rng, 2, 50),
            "min_samples_leaf": randint(rng, 1, 40),
            "max_features": sample_choice(rng, ["sqrt", "log2", 0.35, 0.5, 0.75]),
            "bootstrap": bool(rng.choice([True, False])),
            "class_weight": rng.choice([None, "balanced_subsample"]),
            "n_jobs": -1,
            "random_state": RANDOM_SEED,
        }
        model = RandomForestClassifier(**params)
        return model, params, False
    if model_name == "decision_tree":
        params = {
            "criterion": rng.choice(["gini", "entropy", "log_loss"]),
            "max_depth": rng.choice([None, 3, 4, 5, 6, 8, 10, 12, 16, 20]),
            "min_samples_split": randint(rng, 2, 80),
            "min_samples_leaf": randint(rng, 1, 80),
            "max_features": sample_choice(rng, [None, "sqrt", "log2", 0.35, 0.5, 0.75, 1.0]),
            "class_weight": rng.choice([None, "balanced"]),
            "random_state": RANDOM_SEED,
        }
        model = DecisionTreeClassifier(**params)
        return model, params, False
    if model_name == "extra_trees":
        params = {
            "n_estimators": randint(rng, 30, 120),
            "max_depth": rng.choice([None, 4, 6, 8, 10, 12, 16, 20, 24]),
            "min_samples_split": randint(rng, 2, 50),
            "min_samples_leaf": randint(rng, 1, 40),
            "max_features": sample_choice(rng, ["sqrt", "log2", 0.35, 0.5, 0.75, 1.0]),
            "bootstrap": bool(rng.choice([True, False])),
            "class_weight": rng.choice([None, "balanced"]),
            "n_jobs": -1,
            "random_state": RANDOM_SEED,
        }
        model = ExtraTreesClassifier(**params)
        return model, params, False
    if model_name == "xgboost":
        params = {
            "n_estimators": randint(rng, 80, 280),
            "max_depth": randint(rng, 2, 6),
            "learning_rate": loguniform(rng, 0.015, 0.18),
            "subsample": float(rng.uniform(0.65, 1.0)),
            "colsample_bytree": float(rng.uniform(0.65, 1.0)),
            "min_child_weight": loguniform(rng, 0.7, 25),
            "gamma": loguniform(rng, 1e-4, 3.0),
            "reg_alpha": loguniform(rng, 1e-5, 2.0),
            "reg_lambda": loguniform(rng, 0.1, 25.0),
            "scale_pos_weight": float(ratio * rng.uniform(0.55, 1.45)),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": RANDOM_SEED,
        }
        model = XGBClassifier(**params)
        return model, params, False
    if model_name == "lightgbm":
        params = {
            "n_estimators": randint(rng, 80, 360),
            "learning_rate": loguniform(rng, 0.01, 0.18),
            "num_leaves": randint(rng, 12, 96),
            "max_depth": rng.choice([-1, 3, 4, 5, 6, 8, 10, 12]),
            "min_child_samples": randint(rng, 10, 120),
            "subsample": float(rng.uniform(0.65, 1.0)),
            "colsample_bytree": float(rng.uniform(0.65, 1.0)),
            "reg_alpha": loguniform(rng, 1e-5, 2.0),
            "reg_lambda": loguniform(rng, 0.1, 25.0),
            "class_weight": rng.choice([None, "balanced"]),
            "n_jobs": -1,
            "random_state": RANDOM_SEED,
            "verbosity": -1,
        }
        model = LGBMClassifier(**params)
        return model, params, False
    raise ValueError(f"Unknown model: {model_name}")


def safe_auc(y: pd.Series, prob: np.ndarray) -> float:
    return float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan")


def threshold_table(y: pd.Series, prob: np.ndarray) -> dict[str, Any]:
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), np.quantile(prob, np.linspace(0.01, 0.99, 99))]))
    rows: list[dict[str, float]] = []
    for t in grid:
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = f1_score(y, pred, zero_division=0)
        f2 = fbeta_score(y, pred, beta=2, zero_division=0)
        rows.append(
            {
                "threshold": float(t),
                "sensitivity": float(sens),
                "specificity": float(spec),
                "precision": float(prec),
                "f1": float(f1),
                "f2": float(f2),
                "youden": float(sens + spec - 1),
            }
        )
    by_youden = max(rows, key=lambda r: (r["youden"], r["sensitivity"]))
    by_f2 = max(rows, key=lambda r: (r["f2"], r["sensitivity"]))
    return {"youden": by_youden, "f2": by_f2, "grid": rows}


def metric_block(y: pd.Series, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    metrics = {
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
        "threshold": float(threshold),
        "auroc": safe_auc(y, prob),
        "auprc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    return metrics


def stratified_limit(
    x: pd.DataFrame,
    y: pd.Series,
    max_n: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_n <= 0 or len(y) <= max_n:
        return x, y
    test_size = max_n / len(y)
    _, x_keep, _, y_keep = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if len(np.unique(y)) == 2 else None,
    )
    return x_keep, y_keep


def calibration_slope_intercept(y: pd.Series, prob: np.ndarray) -> dict[str, float]:
    eps = 1e-6
    p = np.clip(prob, eps, 1 - eps)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return {"calibration_intercept": float("nan"), "calibration_slope": float("nan")}
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    cal.fit(x, y)
    return {
        "calibration_intercept": float(cal.intercept_[0]),
        "calibration_slope": float(cal.coef_[0][0]),
    }


def feature_importance(pipeline: Pipeline, model_name: str, top_n: int = 40) -> pd.DataFrame:
    pre = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["model"]
    try:
        names = pre.get_feature_names_out()
    except Exception:
        names = np.array([f"feature_{i}" for i in range(getattr(clf, "n_features_in_", 0))])
    if hasattr(clf, "coef_"):
        values = clf.coef_[0]
        score = np.abs(values)
    elif hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
        score = values
    else:
        return pd.DataFrame(columns=["feature", "importance", "signed_value"])
    n = min(len(names), len(values))
    result = pd.DataFrame(
        {
            "feature": names[:n],
            "importance": score[:n],
            "signed_value": values[:n],
        }
    )
    return result.sort_values("importance", ascending=False).head(top_n)


def optimize_one(
    df: pd.DataFrame,
    outcome_raw: str,
    feature_set: str,
    model_name: str,
    trials: int,
    output_dir: str,
    save_models: bool,
    search_train_max: int,
    search_val_max: int,
) -> dict[str, Any]:
    target_name = PRIMARY_OUTCOMES[outcome_raw]
    target_col = f"target_{target_name}"
    available = set(df.columns)
    categorical, numeric = feature_columns(feature_set, available)
    split = make_split(df, target_col, categorical, numeric)
    seed_material = f"{outcome_raw}|{feature_set}|{model_name}|{RANDOM_SEED}".encode("utf-8")
    stable_seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
    rng = np.random.default_rng(stable_seed)
    trial_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    t0 = time.time()
    x_search_train, y_search_train = stratified_limit(
        split.x_train, split.y_train, search_train_max, stable_seed
    )
    x_search_val, y_search_val = stratified_limit(
        split.x_val, split.y_val, search_val_max, stable_seed + 1
    )
    for trial in range(1, trials + 1):
        model, params, scale = sample_model(model_name, rng, y_search_train)
        preprocess = make_preprocessor(categorical, numeric, scale_numeric=scale)
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        start = time.time()
        try:
            pipe.fit(x_search_train, y_search_train)
            prob = pipe.predict_proba(x_search_val)[:, 1]
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
                best = {**row, "pipeline": pipe, "params_obj": params, "val_prob": prob}
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
        if trial % max(1, min(10, trials // 4)) == 0:
            print(
                f"[{target_name}/{feature_set}/{model_name}] trial {trial}/{trials} "
                f"best_search_auprc={best['search_val_auprc'] if best else None}",
                flush=True,
            )

    if best is None:
        raise RuntimeError(f"All trials failed for {target_name}/{feature_set}/{model_name}")

    best_train_model, _, scale = sample_model(model_name, np.random.default_rng(1), split.y_train)
    best_train_model.set_params(**best["params_obj"])
    best_train_pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(categorical, numeric, scale_numeric=scale)),
            ("model", best_train_model),
        ]
    )
    best_train_pipe.fit(split.x_train, split.y_train)
    full_val_prob = best_train_pipe.predict_proba(split.x_val)[:, 1]
    thresholds = threshold_table(split.y_val, full_val_prob)
    selected_rule = "youden" if target_name == "depressed_mood" else "f2"
    selected_threshold = thresholds[selected_rule]["threshold"]

    # Refit the selected hyperparameters on train+validation, then evaluate on 2025 holdout.
    final_model, _, scale = sample_model(model_name, np.random.default_rng(1), split.y_trainval)
    final_model.set_params(**best["params_obj"])
    final_pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(categorical, numeric, scale_numeric=scale)),
            ("model", final_model),
        ]
    )
    final_pipe.fit(split.x_trainval, split.y_trainval)
    val_metrics = metric_block(split.y_val, full_val_prob, selected_threshold)
    test_prob = final_pipe.predict_proba(split.x_test)[:, 1]
    test_metrics = metric_block(split.y_test, test_prob, selected_threshold)
    test_metrics.update(calibration_slope_intercept(split.y_test, test_prob))

    importances = feature_importance(final_pipe, model_name)
    importance_path = os.path.join(output_dir, f"feature_importance_{target_name}_{feature_set}_{model_name}.csv")
    importances.to_csv(importance_path, index=False, encoding="utf-8-sig")

    trials_path = os.path.join(output_dir, f"trials_{target_name}_{feature_set}_{model_name}.csv")
    pd.DataFrame(trial_rows).to_csv(trials_path, index=False, encoding="utf-8-sig")

    model_path = ""
    if save_models:
        model_path = os.path.join(output_dir, "models", f"model_{target_name}_{feature_set}_{model_name}.joblib")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(final_pipe, model_path)

    result = {
        "outcome": target_name,
        "outcome_raw": outcome_raw,
        "outcome_label": OUTCOME_LABELS[outcome_raw],
        "feature_set": feature_set,
        "model": model_name,
        "trials_requested": trials,
        "trials_completed": int(sum(1 for r in trial_rows if r["status"] == "ok")),
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
        "best_validation_search": {k: v for k, v in best.items() if k not in {"pipeline", "params_obj", "val_prob"}},
        "best_params": best["params_obj"],
        "validation_metrics_at_selected_threshold": val_metrics,
        "test_metrics_at_selected_threshold": test_metrics,
        "feature_importance_path": importance_path,
        "trials_path": trials_path,
        "model_path": model_path,
        "elapsed_seconds": round(time.time() - t0, 3),
        "categorical_features": categorical,
        "numeric_features": numeric,
    }
    return result


def parse_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument(
        "--outcomes",
        default="M_SAD,M_SUI_CON,M_SUI_PLN",
        help="Comma-separated raw outcome variables. Add M_SUI_ATT for secondary attempt model.",
    )
    parser.add_argument("--feature-sets", default="digital_only,standard,full")
    parser.add_argument("--models", default="logistic,sgd_logistic,decision_tree,extra_trees,xgboost,lightgbm")
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--search-train-max", type=int, default=25000)
    parser.add_argument("--search-val-max", type=int, default=12000)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", os.path.join(project_dir(), ".mplconfig"))
    project = project_dir()
    dirs = ensure_dirs(project)
    df, metadata = build_analysis_dataset(project, force=args.force_dataset)
    with open(os.path.join(dirs["outputs"], "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "trials": args.trials,
                "outcomes": parse_list(args.outcomes),
                "feature_sets": parse_list(args.feature_sets),
                "models": parse_list(args.models),
                "metadata": metadata,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for outcome in parse_list(args.outcomes):
        for feature_set in parse_list(args.feature_sets):
            for model_name in parse_list(args.models):
                # Random forests on digital-only are not very informative and can be slow; still run if requested.
                print(f"=== Running {outcome} / {feature_set} / {model_name} ===", flush=True)
                try:
                    result = optimize_one(
                        df=df,
                        outcome_raw=outcome,
                        feature_set=feature_set,
                        model_name=model_name,
                        trials=args.trials,
                        output_dir=dirs["outputs"],
                        save_models=args.save_models,
                        search_train_max=args.search_train_max,
                        search_val_max=args.search_val_max,
                    )
                    results.append(result)
                    pd.DataFrame(results).drop(columns=["best_params"], errors="ignore").to_csv(
                        os.path.join(dirs["outputs"], "model_results_partial.csv"),
                        index=False,
                        encoding="utf-8-sig",
                    )
                    with open(os.path.join(dirs["outputs"], "model_results_partial.json"), "w", encoding="utf-8") as f:
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

    with open(os.path.join(dirs["outputs"], "model_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    if failures:
        with open(os.path.join(dirs["outputs"], "failures.json"), "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2, default=str)

    flat_rows: list[dict[str, Any]] = []
    for r in results:
        row = {
            "outcome": r["outcome"],
            "feature_set": r["feature_set"],
            "model": r["model"],
            "trials_completed": r["trials_completed"],
            "selected_threshold_rule": r["selected_threshold_rule"],
            "selected_threshold": r["selected_threshold"],
            "train_n": r["train_n"],
            "search_train_n": r["search_train_n"],
            "search_validation_n": r["search_validation_n"],
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
        flat_rows.append(row)
    summary = pd.DataFrame(flat_rows).sort_values(["outcome", "feature_set", "test_auprc"], ascending=[True, True, False])
    summary_path = os.path.join(dirs["outputs"], "model_results_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary_path)


if __name__ == "__main__":
    main()
