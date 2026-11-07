from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_yrbss_external_validation as ev


REVISED_SHARED_BASE_FEATURES = [
    "sex",
    "age_years",
    "grade_9_12",
    "academic_achievement_3",
    "body_image_5",
    "bmi_category",
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

REVISED_SHARED_CATEGORICAL_FEATURES = ["sex", "grade_9_12", "bmi_category"]

DIGITAL_FEATURES = {"digital_high_use"}


def bmi_category_fixed_cutoffs(bmi: pd.Series) -> pd.Series:
    x = pd.to_numeric(bmi, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype=object)
    out[x.lt(18.5)] = "underweight"
    out[x.ge(18.5) & x.lt(23.0)] = "normal"
    out[x.ge(23.0) & x.lt(25.0)] = "overweight"
    out[x.ge(25.0)] = "obesity"
    return out


def revised_kyrbs_shared_dataset() -> pd.DataFrame:
    usecols = [
        "YEAR",
        "SEX",
        "AGE",
        "GRADE",
        "academic_achievement_3",
        "PR_BI",
        "bmi_category",
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
        "high_smartphone_4h",
        "target_depressed_mood",
        "target_suicidal_ideation",
        "target_suicide_plan",
        "target_suicide_attempt_secondary",
    ]
    df = pd.read_csv(ev.PROJECT / "data" / "processed" / "kyrbs_2020_2025_lit_reduced_revised_dataset.csv", usecols=usecols, low_memory=False)
    df = df[df["YEAR"].between(2020, 2024)].copy()
    df = df[ev.num(df["GRADE"]).between(3, 6)].copy()

    out = pd.DataFrame(index=df.index)
    out["source"] = "KYRBS"
    out["YEAR"] = ev.num(df["YEAR"])
    sex = pd.Series(np.nan, index=df.index, dtype=object)
    sex[ev.num(df["SEX"]).eq(1)] = "male"
    sex[ev.num(df["SEX"]).eq(2)] = "female"
    out["sex"] = sex
    out["age_years"] = ev.num(df["AGE"])
    out["grade_9_12"] = ev.category_code_string(ev.num(df["GRADE"]) + 6)
    out["academic_achievement_3"] = ev.num(df["academic_achievement_3"])
    out["body_image_5"] = ev.num(df["PR_BI"]).where(ev.num(df["PR_BI"]).between(1, 5), np.nan)
    out["bmi_category"] = df["bmi_category"].astype(object)
    out["physical_activity_days"] = ev.code_minus_one_days(df["PA_TOT"])
    out["physical_activity_5plus"] = ev.binary_from_bool(out["physical_activity_days"] >= 5, out["physical_activity_days"].notna())
    out["breakfast_days"] = ev.code_minus_one_days(df["F_BR"])
    out["trying_to_lose_weight"] = ev.binary_from_bool(ev.num(df["WC_MN"]).eq(2), ev.num(df["WC_MN"]).between(1, 4))
    out["current_alcohol"] = ev.num(df["current_alcohol"])
    out["binge_alcohol_any"] = ev.num(df["heavy_drunk_any"])
    out["lifetime_alcohol_experience"] = ev.num(df["lifetime_alcohol_experience"])
    out["any_current_tobacco"] = ev.num(df["any_current_tobacco"])
    out["lifetime_tobacco_any"] = ev.num(df["lifetime_tobacco_any"])
    out["sexual_experience"] = ev.num(df["sexual_experience"])
    out["sleep_hours"] = ev.num(df["avg_sleep_hours"]).where(ev.num(df["avg_sleep_hours"]).between(2, 16), np.nan)
    out["sleep_8plus"] = ev.binary_from_bool(out["sleep_hours"] >= 8, out["sleep_hours"].notna())
    out["digital_high_use"] = ev.num(df["high_smartphone_4h"])
    out["depressed_mood_for_suicide_models"] = ev.num(df["target_depressed_mood"])

    for target in ev.TARGET_COLS.values():
        out[target] = ev.num(df[target])
    out["sample_weight"] = np.nan
    return out


def revised_yrbss_shared_dataset() -> pd.DataFrame:
    yr = pd.read_csv(ev.YRBSS_DATA, dtype=str, keep_default_na=False, low_memory=False)
    out = pd.DataFrame(index=yr.index)
    out["source"] = "YRBSS"
    out["YEAR"] = 2023
    sex = pd.Series(np.nan, index=yr.index, dtype=object)
    sex[ev.num(yr["Q2"]).eq(1)] = "female"
    sex[ev.num(yr["Q2"]).eq(2)] = "male"
    out["sex"] = sex
    out["age_years"] = ev.num(yr["Q1"]).map({1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 18})
    out["grade_9_12"] = ev.category_code_string(ev.num(yr["Q3"]).map({1: 9, 2: 10, 3: 11, 4: 12}))

    q87 = ev.num(yr["Q87"])
    out["academic_achievement_3"] = np.select([q87.isin([1, 2]), q87.eq(3), q87.isin([4, 5])], [1.0, 2.0, 3.0], default=np.nan)
    out["body_image_5"] = ev.num(yr["Q66"]).where(ev.num(yr["Q66"]).between(1, 5), np.nan)
    height_m = ev.num(yr["Q6"])
    weight_kg = ev.num(yr["Q7"])
    bmi = weight_kg / (height_m**2)
    out["bmi_category"] = bmi_category_fixed_cutoffs(bmi.where(bmi.between(10, 60), np.nan))
    out["physical_activity_days"] = ev.code_minus_one_days(yr["Q76"])
    out["physical_activity_5plus"] = ev.binary_yes1_no2(yr["QN76"])
    out["breakfast_days"] = ev.code_minus_one_days(yr["Q75"])
    out["trying_to_lose_weight"] = ev.binary_from_bool(ev.num(yr["Q67"]).eq(1), ev.num(yr["Q67"]).between(1, 4))
    out["current_alcohol"] = ev.binary_yes1_no2(yr["QN42"])
    out["binge_alcohol_any"] = ev.binary_yes1_no2(yr["QN43"])
    out["lifetime_alcohol_experience"] = ev.binary_from_bool(ev.num(yr["Q41"]).between(2, 7), ev.num(yr["Q41"]).between(1, 7))
    out["any_current_tobacco"] = ev.binary_yes1_no2(yr["qntb4"])
    tobacco_parts = pd.concat([ev.binary_yes1_no2(yr["Q31"]), ev.binary_yes1_no2(yr["QN35"])], axis=1)
    out["lifetime_tobacco_any"] = tobacco_parts.max(axis=1, skipna=True)
    out["sexual_experience"] = ev.binary_yes1_no2(yr["Q56"])
    out["sleep_hours"] = ev.num(yr["Q85"]).map({1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10})
    out["sleep_8plus"] = ev.binary_yes1_no2(yr["QN85"])
    out["digital_high_use"] = ev.binary_yes1_no2(yr["QN80"])

    for target, yr_col in ev.YRBSS_TARGETS.items():
        out[target] = ev.binary_yes1_no2(yr[yr_col])
    out["depressed_mood_for_suicide_models"] = out["target_depressed_mood"]
    out["sample_weight"] = ev.num(yr["weight"])
    out = out[out["grade_9_12"].isin(["9", "10", "11", "12"])].copy()
    return out


def parse_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def configure(kind: str) -> Path:
    out_dir = ev.PROJECT / "outputs" / f"external_validation_yrbss_2023_revised_shared_{kind}"
    ev.BASE_FEATURES = REVISED_SHARED_BASE_FEATURES.copy()
    ev.CATEGORICAL_FEATURES = REVISED_SHARED_CATEGORICAL_FEATURES.copy()
    ev.kyrbs_shared_dataset = revised_kyrbs_shared_dataset
    ev.yrbss_shared_dataset = revised_yrbss_shared_dataset

    def feature_list(outcome_raw: str) -> list[str]:
        features = ev.BASE_FEATURES.copy()
        if kind == "no_digital":
            features = [f for f in features if f not in DIGITAL_FEATURES]
        if outcome_raw != "M_SAD" and kind != "no_depressed_predictor":
            features.append("depressed_mood_for_suicide_models")
        return features

    ev.feature_list = feature_list
    ev.OUT_DIR = out_dir
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["primary", "no_digital", "no_depressed_predictor"], default="primary")
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--outcomes", default="")
    parser.add_argument("--models", default="logistic,sgd_logistic,extra_trees,xgboost,lightgbm")
    parser.add_argument("--search-train-max", type=int, default=25000)
    parser.add_argument("--search-val-max", type=int, default=12000)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = configure(args.kind)
    ev.ensure_dirs()
    default_outcomes = "M_SUI_CON,M_SUI_PLN,M_SUI_ATT" if args.kind == "no_depressed_predictor" else "M_SAD,M_SUI_CON,M_SUI_PLN,M_SUI_ATT"
    outcomes = parse_list(args.outcomes or default_outcomes)
    models = parse_list(args.models)

    kyrbs = ev.kyrbs_shared_dataset()
    yrbss = ev.yrbss_shared_dataset()
    kyrbs.to_csv(out_dir / "tables" / "kyrbs_revised_shared_feature_dataset.csv", index=False, encoding="utf-8-sig")
    yrbss.to_csv(out_dir / "tables" / "yrbss_revised_shared_feature_dataset.csv", index=False, encoding="utf-8-sig")
    ev.save_feature_missingness(kyrbs, yrbss)

    config = {
        "kind": args.kind,
        "trials": args.trials,
        "outcomes": outcomes,
        "models": models,
        "base_features": ev.BASE_FEATURES,
        "categorical_features": ev.CATEGORICAL_FEATURES,
        "harmonization_note": {
            "digital_high_use": "KYRBS high_smartphone_4h vs YRBSS social media several times/day; weak proxy.",
            "bmi_category": "Fixed BMI cutoffs applied to both KYRBS and YRBSS calculated BMI.",
            "unavailable_in_yrbss": ["GAD-7", "loneliness", "weekday smartphone minutes", "weekend smartphone minutes", "high_smartphone_6h", "stress_high", "poor_sleep_recovery"],
        },
        "design": {
            "train": "KYRBS 2020-2022, grade 3-6 (grade 9-12 equivalent)",
            "validation": "KYRBS 2023, threshold selection and hyperparameter selection",
            "internal_temporal_test": "KYRBS 2024",
            "external_test": "YRBSS 2023 national high school",
        },
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    results: list[dict] = []
    partial_path = out_dir / "external_validation_results_partial.json"
    final_path = out_dir / "external_validation_results.json"
    if args.resume:
        load_path = partial_path if partial_path.exists() else final_path
        if load_path.exists():
            results = json.loads(load_path.read_text(encoding="utf-8"))
            print(f"Resuming from {load_path} with {len(results)} completed rows", flush=True)

    completed = {(r["outcome_raw"], r["model"]) for r in results}
    failures = []
    for outcome in outcomes:
        for model_name in models:
            if (outcome, model_name) in completed:
                print(f"=== Skipping completed revised external {args.kind} {outcome} / {model_name} ===", flush=True)
                continue
            print(f"=== Running revised external {args.kind} {outcome} / {model_name} ===", flush=True)
            try:
                result = ev.optimize_external_one(
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
        (out_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    else:
        stale = out_dir / "failures.json"
        if stale.exists():
            stale.unlink()
    ev.write_summary(results, kyrbs, yrbss)
    print(out_dir / "external_validation_results_summary.csv")


if __name__ == "__main__":
    main()
