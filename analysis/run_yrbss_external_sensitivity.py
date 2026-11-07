from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import run_yrbss_external_validation as ev


def parse_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def configure_sensitivity(kind: str) -> Path:
    if kind == "no_depressed_predictor":
        out_dir = ev.PROJECT / "outputs" / "sensitivity_no_depressed_predictor_yrbss_2023"

        def feature_list(outcome_raw: str) -> list[str]:
            return ev.BASE_FEATURES.copy()

        ev.feature_list = feature_list
    elif kind == "no_digital_high_use":
        out_dir = ev.PROJECT / "outputs" / "sensitivity_no_digital_high_use_yrbss_2023"
        ev.BASE_FEATURES = [f for f in ev.BASE_FEATURES if f != "digital_high_use"]
    else:
        raise ValueError(f"Unknown sensitivity kind: {kind}")
    ev.OUT_DIR = out_dir
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["no_depressed_predictor", "no_digital_high_use"], required=True)
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--outcomes", default="")
    parser.add_argument("--models", default="logistic,sgd_logistic,decision_tree,extra_trees,xgboost,lightgbm")
    parser.add_argument("--search-train-max", type=int, default=25000)
    parser.add_argument("--search-val-max", type=int, default=12000)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = configure_sensitivity(args.kind)
    ev.ensure_dirs()
    default_outcomes = "M_SUI_CON,M_SUI_PLN,M_SUI_ATT" if args.kind == "no_depressed_predictor" else "M_SAD,M_SUI_CON,M_SUI_PLN,M_SUI_ATT"
    outcomes = parse_list(args.outcomes or default_outcomes)
    models = parse_list(args.models)

    kyrbs = ev.kyrbs_shared_dataset()
    yrbss = ev.yrbss_shared_dataset()
    kyrbs.to_csv(out_dir / "tables" / "kyrbs_shared_feature_dataset.csv", index=False, encoding="utf-8-sig")
    yrbss.to_csv(out_dir / "tables" / "yrbss_shared_feature_dataset.csv", index=False, encoding="utf-8-sig")
    ev.save_feature_missingness(kyrbs, yrbss)

    config = {
        "trials": args.trials,
        "outcomes": outcomes,
        "models": models,
        "base_features": ev.BASE_FEATURES,
        "categorical_features": ev.CATEGORICAL_FEATURES,
        "sensitivity": args.kind,
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
                print(f"=== Skipping completed {outcome} / {model_name} ===", flush=True)
                continue
            print(f"=== Running {args.kind} external sensitivity {outcome} / {model_name} ===", flush=True)
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
        (out_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        stale = out_dir / "failures.json"
        if stale.exists():
            stale.unlink()
    ev.write_summary(results, kyrbs, yrbss)
    print(out_dir / "external_validation_results_summary.csv")


if __name__ == "__main__":
    main()
