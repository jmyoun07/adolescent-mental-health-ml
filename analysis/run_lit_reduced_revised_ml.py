from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_lit_reduced_ml as ml


OUT_DIR = ml.PROJECT / "outputs" / "ml_lit_reduced_revised_80trials"
DATASET_PATH = ml.PROJECT / "data" / "processed" / "kyrbs_2020_2025_lit_reduced_revised_dataset.csv"
METADATA_PATH = ml.PROJECT / "data" / "processed" / "kyrbs_2020_2025_lit_reduced_revised_metadata.json"

REVISED_BASE_FEATURES = [
    "YEAR",
    "SEX",
    "AGE",
    "GRADE",
    "SCHOOL",
    "CTYPE",
    "academic_achievement_3",
    "economic_status_3",
    "family_structure_3",
    "stress_high",
    "poor_sleep_recovery",
    "avg_sleep_hours",
    "PR_HT",
    "PR_BI",
    "bmi_category",
    "smartphone_weekday_minutes",
    "smartphone_weekend_minutes",
    "high_smartphone_4h",
    "high_smartphone_6h",
    "sedentary_total_avg_minutes",
    "PA_TOT",
    "PA_MSC",
    "current_alcohol",
    "heavy_drunk_any",
    "any_current_tobacco",
    "secondhand_smoke_exposure_any",
    "sexual_experience",
    "loneliness_high",
    "gad7_high10",
    "drug_use_lifetime",
    "F_BR",
    "WC_MN",
    "parent_college_or_higher_any",
    "school_injury_treatment_any",
    "lifetime_alcohol_experience",
    "lifetime_tobacco_any",
    "violence_treatment_any",
    "asthma_lifetime",
    "allergic_rhinitis_lifetime",
    "atopic_dermatitis_lifetime",
    "atopic_disease_any",
    "atopic_disease_count",
    "substance_access_reported_any",
]

REVISED_CATEGORICAL_FEATURES = {
    "SEX",
    "GRADE",
    "SCHOOL",
    "CTYPE",
    "family_structure_3",
    "bmi_category",
}

REMOVED_OR_REPLACED_FEATURES = {
    "stress_3": "replaced by stress_high",
    "sleep_satisfaction_3": "replaced by poor_sleep_recovery",
    "BMI": "replaced by bmi_category",
    "smartphone_avg_daily_minutes": "replaced by smartphone_weekday_minutes, smartphone_weekend_minutes, high_smartphone_4h, high_smartphone_6h",
    "M_SAD_for_suicide_models": "renamed to depressed_mood_binary",
    "loneliness_3_or_M_LON": "replaced by loneliness_high",
    "gad7_total_or_high10": "replaced by gad7_high10",
    "lifetime_cigarette_experience": "removed because it is a subcomponent of lifetime_tobacco_any",
    "gad7_total": "removed; binary GAD-7 >=10 retained",
    "loneliness_3": "removed; binary loneliness_high retained",
    "atopic_disease_count": "retained as sensitivity-style comorbidity burden",
}


def clean_sentinel(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([7777, 8888, 9999], np.nan)


def to_string_frame(x):
    return x.astype(str)


def bmi_category_fixed_cutoffs(bmi: pd.Series) -> pd.Series:
    x = pd.to_numeric(bmi, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype=object)
    out[x.lt(18.5)] = "underweight"
    out[x.ge(18.5) & x.lt(23.0)] = "normal"
    out[x.ge(23.0) & x.lt(25.0)] = "overweight"
    out[x.ge(25.0)] = "obesity"
    return out


def build_revised_dataset(force: bool = False) -> tuple[pd.DataFrame, dict]:
    if DATASET_PATH.exists() and METADATA_PATH.exists() and not force:
        data = pd.read_csv(DATASET_PATH, low_memory=False)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return data, metadata

    base_lit, old_meta = ml.build_lit_dataset(force=False)
    raw_needed = [
        "M_STR",
        "M_SLP_EN",
        "smartphone_weekday_minutes",
        "smartphone_weekend_minutes",
        "high_smartphone_4h",
        "high_smartphone_6h",
        "BMI",
    ]
    raw = pd.read_csv(ml.PROCESSED, usecols=raw_needed, low_memory=False)
    if len(base_lit) != len(raw):
        raise RuntimeError(f"Row mismatch: lit={len(base_lit)} raw={len(raw)}")

    df = base_lit.copy()
    m_str = clean_sentinel(raw["M_STR"])
    m_slp_en = clean_sentinel(raw["M_SLP_EN"])
    df["stress_high"] = np.where(m_str.notna(), m_str.isin([1, 2]).astype(float), np.nan)
    df["poor_sleep_recovery"] = np.where(m_slp_en.notna(), m_slp_en.isin([4, 5]).astype(float), np.nan)
    df["depressed_mood_binary"] = df["target_depressed_mood"]
    df["smartphone_weekday_minutes"] = pd.to_numeric(raw["smartphone_weekday_minutes"], errors="coerce")
    df["smartphone_weekend_minutes"] = pd.to_numeric(raw["smartphone_weekend_minutes"], errors="coerce")
    df["high_smartphone_4h"] = pd.to_numeric(raw["high_smartphone_4h"], errors="coerce")
    df["high_smartphone_6h"] = pd.to_numeric(raw["high_smartphone_6h"], errors="coerce")
    df["bmi_category"] = bmi_category_fixed_cutoffs(raw["BMI"])

    keep_cols = [
        "YEAR",
        "target_depressed_mood",
        "target_suicidal_ideation",
        "target_suicide_plan",
        "target_suicide_attempt_secondary",
        *REVISED_BASE_FEATURES,
        "depressed_mood_binary",
    ]
    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in df.columns]))
    revised = df[keep_cols].copy()

    missing = [f for f in [*REVISED_BASE_FEATURES, "depressed_mood_binary"] if f not in revised.columns]
    metadata = {
        "rows": int(len(revised)),
        "source_dataset": str(ml.PROJECT / "data" / "processed" / "kyrbs_2020_2025_lit_reduced_dataset.csv"),
        "revised_dataset": str(DATASET_PATH),
        "feature_set": "lit_revised_full_2020_2024",
        "features_base": REVISED_BASE_FEATURES,
        "suicide_outcome_extra_feature": "depressed_mood_binary",
        "missing_revised_features": missing,
        "removed_or_replaced_features": REMOVED_OR_REPLACED_FEATURES,
        "coding_rules": {
            "depressed_mood_binary": "KYRBS M_SAD recoded as 1 for depressed mood/sadness-hopelessness experience; this is a symptom screen, not a clinical depression diagnosis.",
            "gad7_high10": "GAD-7 total score >=10, a common cutoff for clinically meaningful/moderate anxiety symptoms.",
            "loneliness_high": "M_LON high-frequency loneliness, coded as original categories 4 or 5.",
            "stress_high": "M_STR original categories 1 or 2, high/very high perceived stress.",
            "poor_sleep_recovery": "M_SLP_EN original categories 4 or 5, insufficient/very insufficient sleep recovery.",
            "bmi_category": "Fixed BMI cutoffs: underweight <18.5, normal 18.5-<23, overweight 23-<25, obesity >=25. For a final pediatric epidemiology paper, age/sex percentile-based sensitivity is recommended.",
            "smartphone_weekday_minutes": "Original weekday smartphone average minutes.",
            "smartphone_weekend_minutes": "Original weekend smartphone average minutes.",
            "high_smartphone_4h": "Average daily smartphone use >=240 minutes, retained as a digital phenotype threshold feature.",
            "high_smartphone_6h": "Average daily smartphone use >=360 minutes, retained as a digital phenotype threshold feature.",
        },
        "old_metadata": old_meta,
    }
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    revised.to_csv(DATASET_PATH, index=False, encoding="utf-8-sig")
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return revised, metadata


def feature_columns_revised(df: pd.DataFrame, feature_set: str, outcome_raw: str):
    if feature_set != "lit_full_2020_2024":
        raise ValueError("This revised analysis uses lit_full_2020_2024 temporal split only.")
    requested = REVISED_BASE_FEATURES.copy()
    if outcome_raw != "M_SAD":
        requested.append("depressed_mood_binary")
    available = [f for f in requested if f in df.columns]
    categorical = [f for f in available if f in REVISED_CATEGORICAL_FEATURES]
    numeric = [f for f in available if f not in REVISED_CATEGORICAL_FEATURES]
    return available, categorical, numeric


def parse_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def write_summary(results: list[dict]) -> None:
    rows = []
    for r in results:
        row = {
            "outcome": r["outcome"],
            "outcome_raw": r["outcome_raw"],
            "feature_set": "lit_revised_full_2020_2024",
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
    summary = pd.DataFrame(rows).sort_values(["outcome", "test_auprc", "test_auroc"], ascending=[True, False, False])
    summary.to_csv(OUT_DIR / "model_results_summary.csv", index=False, encoding="utf-8-sig")

    primary = summary.sort_values(["outcome", "search_val_auprc", "search_val_auroc"], ascending=[True, False, False])
    primary = primary.groupby("outcome", as_index=False).head(1)
    primary.to_csv(OUT_DIR / "primary_model_by_validation.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--outcomes", default="M_SAD,M_SUI_CON,M_SUI_PLN,M_SUI_ATT")
    parser.add_argument("--models", default="logistic,sgd_logistic,extra_trees,xgboost,lightgbm")
    parser.add_argument("--search-train-max", type=int, default=25000)
    parser.add_argument("--search-val-max", type=int, default=12000)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-dataset", action="store_true")
    args = parser.parse_args()

    ml.OUT_DIR = OUT_DIR
    ml.CATEGORICAL_FEATURES = REVISED_CATEGORICAL_FEATURES
    ml.feature_columns = feature_columns_revised
    ml.ensure_dirs()

    df, metadata = build_revised_dataset(force=args.force_dataset)
    config = {
        "trials": args.trials,
        "outcomes": parse_list(args.outcomes),
        "feature_sets": ["lit_revised_full_2020_2024"],
        "models": parse_list(args.models),
        "optimizer": "80-trial seeded randomized hyperparameter search; Optuna is not installed in .venv_ml.",
        "threshold_selection": "Depressed mood uses Youden threshold on validation; suicidality outcomes use F2 threshold.",
        "metadata": metadata,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    results: list[dict] = []
    partial_path = OUT_DIR / "model_results_partial.json"
    final_path = OUT_DIR / "model_results.json"
    if args.resume:
        load_path = partial_path if partial_path.exists() else final_path
        if load_path.exists():
            results = json.loads(load_path.read_text(encoding="utf-8"))
            print(f"Resuming from {load_path} with {len(results)} completed rows", flush=True)

    completed = {(r["outcome_raw"], r["model"]) for r in results}
    failures = []
    for outcome in parse_list(args.outcomes):
        for model_name in parse_list(args.models):
            if (outcome, model_name) in completed:
                print(f"=== Skipping completed revised {outcome} / {model_name} ===", flush=True)
                continue
            print(f"=== Running revised lit-reduced {outcome} / {model_name} ===", flush=True)
            try:
                result = ml.optimize_one(
                    df=df,
                    outcome_raw=outcome,
                    feature_set="lit_full_2020_2024",
                    model_name=model_name,
                    trials=args.trials,
                    save_models=args.save_models,
                    search_train_max=args.search_train_max,
                    search_val_max=args.search_val_max,
                )
                result["feature_set"] = "lit_revised_full_2020_2024"
                results.append(result)
                partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            except Exception as exc:
                failure = {"outcome": outcome, "model": model_name, "error": repr(exc)}
                failures.append(failure)
                print(f"FAILED {failure}", flush=True)

    final_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if failures:
        (OUT_DIR / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    else:
        stale = OUT_DIR / "failures.json"
        if stale.exists():
            stale.unlink()
    write_summary(results)
    print(OUT_DIR / "model_results_summary.csv")


if __name__ == "__main__":
    main()
