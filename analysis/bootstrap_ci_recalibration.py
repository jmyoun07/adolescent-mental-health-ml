from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "outputs" / "ci_recalibration"


def calibration_slope_intercept_fast(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    x = logit(np.clip(prob, eps, 1 - eps)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    cal.fit(x, y)
    return float(cal.intercept_[0]), float(cal.coef_[0][0])


def metric_values(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, prob)),
        "auprc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "calibration_intercept": calibration_slope_intercept_fast(y, prob)[0],
        "calibration_slope": calibration_slope_intercept_fast(y, prob)[1],
    }


def bootstrap_ci(y: np.ndarray, prob: np.ndarray, n_boot: int, seed: int) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(y)
    rows = []
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yi = y[idx]
        if len(np.unique(yi)) < 2:
            continue
        pi = prob[idx]
        row = {"bootstrap": i + 1}
        row.update(metric_values(yi, pi))
        rows.append(row)
    return rows


def ci_summary(rows: list[dict[str, float]], point: dict[str, float]) -> list[dict[str, float]]:
    df = pd.DataFrame(rows)
    out = []
    for metric in ["auroc", "auprc", "brier", "calibration_intercept", "calibration_slope"]:
        out.append(
            {
                "metric": metric,
                "estimate": point[metric],
                "ci_low": float(df[metric].quantile(0.025)),
                "ci_high": float(df[metric].quantile(0.975)),
                "bootstrap_n": int(df[metric].notna().sum()),
            }
        )
    return out


def intercept_only_recalibration(y: np.ndarray, prob: np.ndarray) -> np.ndarray:
    eps = 1e-6
    lp = logit(np.clip(prob, eps, 1 - eps))

    def score(a: float) -> float:
        return float(expit(lp + a).mean() - y.mean())

    intercept = brentq(score, -30, 30)
    return expit(lp + intercept)


def slope_intercept_recalibration(y: np.ndarray, prob: np.ndarray) -> np.ndarray:
    eps = 1e-6
    x = logit(np.clip(prob, eps, 1 - eps)).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    cal.fit(x, y)
    return cal.predict_proba(x)[:, 1]


def load_probability(dataset: str, outcome: str, model: str) -> pd.DataFrame:
    if dataset == "kyrbs_internal_2024":
        path = (
            PROJECT
            / "outputs"
            / "ml_lit_reduced_80trials"
            / "probabilities"
            / f"test_probabilities_{outcome}_lit_full_2020_2024_{model}.csv"
        )
    elif dataset == "yrbss_external_2023":
        path = (
            PROJECT
            / "outputs"
            / "external_validation_yrbss_2023"
            / "probabilities"
            / f"yrbss_probabilities_{outcome}_{model}.csv"
        )
    else:
        raise ValueError(dataset)
    df = pd.read_csv(path)
    return df.rename(columns={"probability": "prob"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kyrbs_primary = pd.read_csv(PROJECT / "outputs" / "ml_lit_reduced_80trials" / "main_2020_2024_best_models.csv")
    yrbss_primary = pd.read_csv(PROJECT / "outputs" / "external_validation_yrbss_2023" / "primary_model_by_kyrbs_validation.csv")
    specs = []
    for _, row in kyrbs_primary.iterrows():
        specs.append({"dataset": "kyrbs_internal_2024", "outcome": row["outcome"], "model": row["model"]})
    for _, row in yrbss_primary.iterrows():
        specs.append({"dataset": "yrbss_external_2023", "outcome": row["outcome"], "model": row["model"]})

    point_rows = []
    ci_rows = []
    recal_rows = []
    for spec in specs:
        df = load_probability(spec["dataset"], spec["outcome"], spec["model"])
        y = df["y_true"].astype(int).to_numpy()
        prob = df["prob"].astype(float).to_numpy()
        point = metric_values(y, prob)
        point_rows.append({**spec, **point, "n": len(y), "events": int(y.sum()), "prevalence": float(y.mean())})
        boot = bootstrap_ci(y, prob, args.n_boot, args.seed + abs(hash((spec["dataset"], spec["outcome"])) % 100000))
        pd.DataFrame(boot).to_csv(
            OUT_DIR / f"bootstrap_{spec['dataset']}_{spec['outcome']}_{spec['model']}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        for row in ci_summary(boot, point):
            ci_rows.append({**spec, **row})

        if spec["dataset"] == "yrbss_external_2023":
            for method, recal_prob in [
                ("original", prob),
                ("intercept_only", intercept_only_recalibration(y, prob)),
                ("intercept_and_slope", slope_intercept_recalibration(y, prob)),
            ]:
                vals = metric_values(y, recal_prob)
                recal_rows.append({**spec, "recalibration": method, **vals})

    pd.DataFrame(point_rows).to_csv(OUT_DIR / "point_estimates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ci_rows).to_csv(OUT_DIR / "bootstrap_ci_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(recal_rows).to_csv(OUT_DIR / "yrbss_recalibration_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR / "bootstrap_ci_summary.csv")
    print(OUT_DIR / "yrbss_recalibration_summary.csv")


if __name__ == "__main__":
    main()
