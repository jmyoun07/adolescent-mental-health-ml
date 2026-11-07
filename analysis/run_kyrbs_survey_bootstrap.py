from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "outputs" / "remaining_methodology_fixes_2026_07_09"
DOMESTIC_DIR = PROJECT / "outputs" / "ml_lit_reduced_revised_80trials"
REVISED = PROJECT / "data" / "processed" / "kyrbs_2020_2025_lit_reduced_revised_dataset.csv"
PROTOCOL = PROJECT / "data" / "processed" / "kyrbs_2020_2025_protocol_analysis_dataset.csv"

TARGET_COLS = {
    "depressed_mood": "target_depressed_mood",
    "suicidal_ideation": "target_suicidal_ideation",
    "suicide_plan": "target_suicide_plan",
    "suicide_attempt_secondary": "target_suicide_attempt_secondary",
}


def metrics(y: np.ndarray, prob: np.ndarray, weight: np.ndarray | None = None) -> dict[str, float]:
    if weight is not None:
        valid = np.isfinite(weight) & (weight > 0) & np.isfinite(prob)
        y = y[valid]
        prob = prob[valid]
        weight = weight[valid]
    out = {
        "n": int(len(y)),
        "events": int(np.sum(y)),
        "prevalence": float(np.average(y, weights=weight)) if weight is not None else float(np.mean(y)),
        "mean_predicted_risk": float(np.average(prob, weights=weight))
        if weight is not None
        else float(np.mean(prob)),
    }
    if len(np.unique(y)) < 2:
        out.update(
            {
                "auroc": np.nan,
                "auprc": np.nan,
                "brier": np.nan,
                "calibration_intercept": np.nan,
                "calibration_slope": np.nan,
            }
        )
        return out
    out["auroc"] = float(roc_auc_score(y, prob, sample_weight=weight))
    out["auprc"] = float(average_precision_score(y, prob, sample_weight=weight))
    out["brier"] = (
        float(np.average((y - prob) ** 2, weights=weight))
        if weight is not None
        else float(brier_score_loss(y, prob))
    )
    x = logit(np.clip(prob, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    cal.fit(x, y, sample_weight=weight)
    out["calibration_intercept"] = float(cal.intercept_[0])
    out["calibration_slope"] = float(cal.coef_[0][0])
    return out


def percentile_ci(values: pd.Series) -> tuple[float, float]:
    values = values.dropna()
    if values.empty:
        return np.nan, np.nan
    return float(values.quantile(0.025)), float(values.quantile(0.975))


def stratified_cluster_indices(design: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    sampled: list[np.ndarray] = []
    indexed = design.reset_index().rename(columns={"index": "_row_index"})
    for _, s_df in indexed.groupby("STRATA", dropna=False):
        clusters = s_df["CLUSTER"].dropna().unique()
        if len(clusters) == 0:
            sampled.append(s_df["_row_index"].to_numpy())
            continue
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        groups = {cluster: g["_row_index"].to_numpy() for cluster, g in s_df.groupby("CLUSTER", dropna=False)}
        for cluster in sampled_clusters:
            sampled.append(groups[cluster])
    return np.concatenate(sampled).astype(int)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_bootstrap = 500
    rng = np.random.default_rng(20260711)
    revised = pd.read_csv(REVISED, usecols=["YEAR", *TARGET_COLS.values()], low_memory=False)
    protocol = pd.read_csv(PROTOCOL, usecols=["YEAR", "STRATA", "CLUSTER", "WT"], low_memory=False)
    if len(revised) != len(protocol):
        raise RuntimeError(f"Row mismatch: revised={len(revised)} protocol={len(protocol)}")
    if not (pd.to_numeric(revised["YEAR"], errors="coerce").fillna(-1).astype(int).to_numpy() == pd.to_numeric(protocol["YEAR"], errors="coerce").fillna(-1).astype(int).to_numpy()).all():
        raise RuntimeError("YEAR alignment mismatch between revised and protocol datasets")

    specs = pd.read_csv(DOMESTIC_DIR / "primary_model_by_validation.csv")
    iterations = []
    point_rows = []
    for _, spec in specs.iterrows():
        outcome = spec["outcome"]
        model = spec["model"]
        target = TARGET_COLS[outcome]
        mask = revised["YEAR"].eq(2024) & revised[target].isin([0, 1])
        design = protocol.loc[mask, ["STRATA", "CLUSTER", "WT"]].reset_index(drop=True)
        prob_path = DOMESTIC_DIR / "probabilities" / f"test_probabilities_{outcome}_lit_full_2020_2024_{model}.csv"
        prob_df = pd.read_csv(prob_path).rename(columns={"probability": "prob"})
        if len(prob_df) != len(design):
            raise RuntimeError(f"Length mismatch for {outcome}: prob={len(prob_df)} design={len(design)}")
        y = prob_df["y_true"].astype(int).to_numpy()
        prob = prob_df["prob"].astype(float).to_numpy()
        weight = pd.to_numeric(design["WT"], errors="coerce").astype(float).to_numpy()
        point_rows.append(
            {
                "outcome": outcome,
                "model": model,
                "bootstrap_type": "kyrbs_stratified_cluster_weighted",
                **metrics(y, prob, weight),
                "strata_n": int(design["STRATA"].nunique(dropna=True)),
                "cluster_n": int(design["CLUSTER"].nunique(dropna=True)),
            }
        )
        for b in range(n_bootstrap):
            idx = stratified_cluster_indices(design, rng)
            yy = y[idx]
            if len(np.unique(yy)) < 2:
                continue
            iterations.append(
                {
                    "outcome": outcome,
                    "model": model,
                    "bootstrap_iteration": b + 1,
                    **metrics(yy, prob[idx], weight[idx]),
                }
            )

    iter_df = pd.DataFrame(iterations)
    point = pd.DataFrame(point_rows)
    rows = []
    for (outcome, model), g in iter_df.groupby(["outcome", "model"]):
        p = point[(point["outcome"] == outcome) & (point["model"] == model)].iloc[0].to_dict()
        for metric in ["auroc", "auprc", "brier", "calibration_intercept", "calibration_slope"]:
            low, high = percentile_ci(g[metric])
            rows.append(
                {
                    "outcome": outcome,
                    "model": model,
                    "metric": metric,
                    "point_estimate": p[metric],
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_valid_n": int(g[metric].notna().sum()),
                    "bootstrap_type": "kyrbs_stratified_cluster_weighted",
                    "strata_n": p["strata_n"],
                    "cluster_n": p["cluster_n"],
                }
            )
    summary = pd.DataFrame(rows)
    point.to_csv(OUT_DIR / "kyrbs_survey_design_point_estimates.csv", index=False, encoding="utf-8-sig")
    iter_df.to_csv(OUT_DIR / "kyrbs_survey_cluster_bootstrap_iterations.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "kyrbs_survey_cluster_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    print(OUT_DIR / "kyrbs_survey_cluster_bootstrap_ci.csv")


if __name__ == "__main__":
    main()
