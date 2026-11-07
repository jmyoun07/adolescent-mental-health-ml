from __future__ import annotations

import argparse
import json

import pandas as pd

import run_lit_reduced_ml as ml


OUT_DIR = ml.PROJECT / "outputs" / "sensitivity_behavior_only_kyrbs_2020_2024"

DISTRESS_FEATURES = {
    "stress_3",
    "sleep_satisfaction_3",
    "M_SAD_for_suicide_models",
    "loneliness_3_or_M_LON",
    "gad7_total_or_high10",
    "gad7_total",
    "gad7_high10",
    "loneliness_3",
    "loneliness_high",
}


def ensure_dirs() -> None:
    for sub in ["", "models", "feature_importance", "trials", "probabilities"]:
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def patch_feature_loader() -> None:
    original_loader = ml.load_decision_features

    def load_behavior_only_features(feature_set: str, outcome_raw: str) -> list[str]:
        features = original_loader(feature_set, outcome_raw)
        return [f for f in features if f not in DISTRESS_FEATURES]

    ml.load_decision_features = load_behavior_only_features


def write_summary(results: list[dict]) -> None:
    rows = []
    for r in results:
        row = {
            "outcome": r["outcome"],
            "outcome_raw": r["outcome_raw"],
            "feature_set": r["feature_set"],
            "model": r["model"],
            "trials_completed": r["trials_completed"],
            "features_n": r["features_n"],
            "excluded_features": ",".join(sorted(DISTRESS_FEATURES)),
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


def parse_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


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

    ml.OUT_DIR = OUT_DIR
    ensure_dirs()
    patch_feature_loader()
    df, metadata = ml.build_lit_dataset(force=False)

    config = {
        "trials": args.trials,
        "outcomes": parse_list(args.outcomes),
        "feature_sets": ["lit_full_2020_2024"],
        "models": parse_list(args.models),
        "sensitivity": "Behavior-only KYRBS enhanced model excluding concurrent/near-concurrent distress predictors.",
        "excluded_features": sorted(DISTRESS_FEATURES),
        "metadata": metadata,
    }
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
                print(f"=== Skipping completed behavior-only KYRBS {outcome} / {model_name} ===", flush=True)
                continue
            print(f"=== Running behavior-only KYRBS sensitivity {outcome} / {model_name} ===", flush=True)
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
        stale = OUT_DIR / "failures.json"
        if stale.exists():
            stale.unlink()
    write_summary(results)
    print(OUT_DIR / "model_results_summary.csv")


if __name__ == "__main__":
    main()
