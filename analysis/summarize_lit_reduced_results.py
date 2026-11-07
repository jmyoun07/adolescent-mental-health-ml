from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "outputs" / "ml_lit_reduced_80trials"
REPORTS = PROJECT / "reports"


def top_features(outcome: str, feature_set: str, model: str, n: int = 8) -> str:
    path = OUT_DIR / "feature_importance" / f"feature_importance_{outcome}_{feature_set}_{model}.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    if df.empty:
        return ""
    return "; ".join(df["feature"].head(n).astype(str).tolist())


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(OUT_DIR / "model_results_summary.csv")
    best_feature = summary.loc[summary.groupby(["outcome", "feature_set"])["test_auprc"].idxmax()].copy()
    best_overall = summary.loc[summary.groupby("outcome")["test_auprc"].idxmax()].copy()

    cols = [
        "outcome",
        "feature_set",
        "model",
        "features_n",
        "trials_completed",
        "test_n",
        "test_prevalence",
        "test_auroc",
        "test_auprc",
        "test_brier",
        "selected_threshold_rule",
        "selected_threshold",
        "test_sensitivity",
        "test_specificity",
        "test_precision",
        "test_f1",
        "test_f2",
        "test_calibration_intercept",
        "test_calibration_slope",
    ]
    best_feature[cols].sort_values(["outcome", "feature_set"]).to_csv(
        OUT_DIR / "best_model_by_outcome_feature_set.csv", index=False, encoding="utf-8-sig"
    )
    best_overall[cols].sort_values("outcome").to_csv(
        OUT_DIR / "best_model_by_outcome_overall.csv", index=False, encoding="utf-8-sig"
    )

    lines: list[str] = []
    lines.append("# Literature-Reduced KYRBS ML Run Report")
    lines.append("")
    lines.append("## Run Scope")
    lines.append("")
    lines.append("- Outcomes: depressed mood, suicidal ideation, suicide plan, suicide attempt.")
    lines.append("- Models: logistic, sgd_logistic, decision_tree, extra_trees, xgboost, lightgbm.")
    lines.append("- Trials: 80 random-search trials for every outcome, feature set, and model combination.")
    lines.append("- Feature sets:")
    lines.append("  - `lit_compatible_2020_2025`: 2020-2023 train, 2024 validation, 2025 holdout; excludes variables not observed in 2025.")
    lines.append("  - `lit_full_2020_2024`: 2020-2022 train, 2023 validation, 2024 holdout; includes GAD-7, loneliness, violence, and atopic disease variables unavailable in 2025.")
    lines.append("")
    lines.append("## Completion Check")
    lines.append("")
    lines.append(f"- Result rows: {len(summary)}")
    lines.append(f"- Minimum completed trials per row: {int(summary['trials_completed'].min())}")
    lines.append(f"- Maximum completed trials per row: {int(summary['trials_completed'].max())}")
    lines.append("")
    lines.append("## Best Models by Outcome and Feature Set")
    lines.append("")
    table_cols = [
        "outcome",
        "feature_set",
        "model",
        "features_n",
        "test_prevalence",
        "test_auroc",
        "test_auprc",
        "test_brier",
        "selected_threshold",
        "test_sensitivity",
        "test_specificity",
        "test_precision",
    ]
    table = best_feature[table_cols].sort_values(["outcome", "feature_set"]).copy()
    for c in [
        "test_prevalence",
        "test_auroc",
        "test_auprc",
        "test_brier",
        "selected_threshold",
        "test_sensitivity",
        "test_specificity",
        "test_precision",
    ]:
        table[c] = table[c].map(lambda x: f"{x:.3f}")
    lines.append(markdown_table(table))
    lines.append("")
    lines.append("## Primary vs Supplemental Interpretation")
    lines.append("")
    lines.append("- Primary internal temporal-holdout results should use `lit_compatible_2020_2025`, because those predictors are available through the 2025 holdout year.")
    lines.append("- `lit_full_2020_2024` generally performs better because it includes GAD-7 and loneliness, but it cannot be evaluated on 2025 and should be framed as an enhanced/sensitivity analysis.")
    lines.append("- Suicide outcomes use F2-derived thresholds to prioritize sensitivity; depressed mood uses Youden threshold.")
    lines.append("- The run does not yet implement YRBSS external validation, survey-weighted metrics, decision-curve analysis, NRI/IDI, or bootstrap confidence intervals.")
    lines.append("")
    lines.append("## Top Features for Best Models")
    lines.append("")
    for _, row in best_feature.sort_values(["outcome", "feature_set"]).iterrows():
        features = top_features(row["outcome"], row["feature_set"], row["model"])
        lines.append(f"- {row['outcome']} / {row['feature_set']} / {row['model']}: {features}")
    lines.append("")
    lines.append("## Key Output Files")
    lines.append("")
    lines.append(f"- Summary: `{OUT_DIR / 'model_results_summary.csv'}`")
    lines.append(f"- Best by outcome and feature set: `{OUT_DIR / 'best_model_by_outcome_feature_set.csv'}`")
    lines.append(f"- Best overall: `{OUT_DIR / 'best_model_by_outcome_overall.csv'}`")
    lines.append(f"- Variable availability check: `{REPORTS / 'lit_reduced_variable_year_availability_check.csv'}`")
    lines.append("")
    (REPORTS / "lit_reduced_ml_run_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(REPORTS / "lit_reduced_ml_run_report.md")


if __name__ == "__main__":
    main()
