from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from sklearn.pipeline import Pipeline

import run_lit_reduced_ml as ml
import run_yrbss_external_validation as ev
from run_lit_reduced_ml import normalize_params
from run_protocol_ml import sample_model

PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "outputs" / "shap_tree_contributions"

OUTCOME_RAW = {
    "depressed_mood": "M_SAD",
    "suicidal_ideation": "M_SUI_CON",
    "suicide_plan": "M_SUI_PLN",
    "suicide_attempt_secondary": "M_SUI_ATT",
}


def ensure_dirs() -> None:
    for sub in ["", "tables"]:
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def stratified_sample(x: pd.DataFrame, y: pd.Series, max_n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    if len(y) <= max_n:
        return x, y
    rng = np.random.default_rng(seed)
    keep = []
    arr = y.to_numpy()
    for cls in np.unique(arr):
        idx = np.flatnonzero(arr == cls)
        n_cls = max(1, int(round(max_n * len(idx) / len(arr))))
        n_cls = min(n_cls, len(idx))
        keep.extend(rng.choice(idx, size=n_cls, replace=False).tolist())
    keep = np.array(sorted(keep))
    return x.iloc[keep], y.iloc[keep]


def base_feature_name(name: str, raw_features: list[str], categorical: list[str]) -> str:
    clean = name
    for prefix in ["num__", "cat__"]:
        if clean.startswith(prefix):
            clean = clean[len(prefix) :]
    if clean.startswith("missingindicator_"):
        clean = clean[len("missingindicator_") :]
    for cat in categorical:
        if clean == cat or clean.startswith(cat + "_"):
            return cat
    for feature in sorted(raw_features, key=len, reverse=True):
        if clean == feature or clean.startswith(feature + "_"):
            return feature
    return clean


def tree_contributions(pipe: Pipeline, x: pd.DataFrame, raw_features: list[str], categorical: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    xt = pre.transform(x)
    names = list(pre.get_feature_names_out())
    if hasattr(model, "get_booster"):
        dm = xgb.DMatrix(xt, feature_names=names)
        contrib = model.get_booster().predict(dm, pred_contribs=True)
    elif model.__class__.__name__.lower().startswith("lgbm"):
        contrib = model.predict(xt, pred_contrib=True)
    else:
        raise ValueError(f"Unsupported model for tree SHAP contributions: {type(model)}")

    if contrib.ndim == 3:
        contrib = contrib[:, :, 1]
    contrib = np.asarray(contrib)
    feature_contrib = contrib[:, : len(names)]
    mean_abs = np.abs(feature_contrib).mean(axis=0)
    signed_mean = feature_contrib.mean(axis=0)
    transformed = pd.DataFrame(
        {
            "transformed_feature": names,
            "base_feature": [base_feature_name(n, raw_features, categorical) for n in names],
            "mean_abs_shap": mean_abs,
            "mean_shap": signed_mean,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    base = (
        transformed.groupby("base_feature", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"), mean_shap=("mean_shap", "sum"))
        .sort_values("mean_abs_shap", ascending=False)
    )
    return transformed, base


def fit_kyrbs_primary_model(outcome: str, model_name: str) -> tuple[Pipeline, pd.DataFrame, pd.Series, list[str], list[str]]:
    results = json.loads((PROJECT / "outputs" / "ml_lit_reduced_80trials" / "model_results.json").read_text(encoding="utf-8"))
    outcome_raw = OUTCOME_RAW[outcome]
    row = next(
        r
        for r in results
        if r["outcome"] == outcome and r["feature_set"] == "lit_full_2020_2024" and r["model"] == model_name
    )
    df, _ = ml.build_lit_dataset(force=False)
    features, categorical, numeric = ml.feature_columns(df, "lit_full_2020_2024", outcome_raw)
    target_col = f"target_{outcome}"
    split = ml.make_split(df, target_col, "lit_full_2020_2024", features)
    model, _, scale = sample_model(model_name, np.random.default_rng(1), split.y_trainval)
    model, _ = normalize_params(model, row["best_params"])
    pipe = Pipeline(
        [
            ("preprocess", ml.make_preprocessor(categorical, numeric, scale_numeric=scale)),
            ("model", model),
        ]
    )
    pipe.fit(split.x_trainval, split.y_trainval)
    x_sample, y_sample = stratified_sample(split.x_test[features], split.y_test, 5000, 20260707)
    return pipe, x_sample, y_sample, features, categorical


def load_yrbss_primary_model(outcome: str, model_name: str) -> tuple[Pipeline, pd.DataFrame, pd.Series, list[str], list[str]]:
    outcome_raw = OUTCOME_RAW[outcome]
    pipe = joblib.load(PROJECT / "outputs" / "external_validation_yrbss_2023" / "models" / f"model_{outcome}_{model_name}.joblib")
    yrbss = ev.yrbss_shared_dataset()
    target_col = ev.TARGET_COLS[outcome_raw]
    external = yrbss[yrbss[target_col].isin([0, 1])].copy()
    features = ev.feature_list(outcome_raw)
    categorical = [f for f in features if f in ev.CATEGORICAL_FEATURES]
    x_sample, y_sample = stratified_sample(external[features], external[target_col].astype(int), 5000, 20260707)
    return pipe, x_sample, y_sample, features, categorical


def main() -> None:
    ensure_dirs()
    kyrbs_primary = pd.read_csv(PROJECT / "outputs" / "ml_lit_reduced_80trials" / "main_2020_2024_best_models.csv")
    yrbss_primary = pd.read_csv(PROJECT / "outputs" / "external_validation_yrbss_2023" / "primary_model_by_kyrbs_validation.csv")
    all_base = []
    for analysis, primary, loader in [
        ("kyrbs_internal_2024", kyrbs_primary, fit_kyrbs_primary_model),
        ("yrbss_external_2023", yrbss_primary, load_yrbss_primary_model),
    ]:
        for _, row in primary.iterrows():
            outcome = row["outcome"]
            model = row["model"]
            if model not in {"xgboost", "lightgbm"}:
                continue
            print(f"Computing tree SHAP contributions: {analysis} / {outcome} / {model}", flush=True)
            pipe, x_sample, y_sample, features, categorical = loader(outcome, model)
            transformed, base = tree_contributions(pipe, x_sample, features, categorical)
            transformed.insert(0, "model", model)
            transformed.insert(0, "outcome", outcome)
            transformed.insert(0, "analysis", analysis)
            base.insert(0, "model", model)
            base.insert(0, "outcome", outcome)
            base.insert(0, "analysis", analysis)
            transformed.to_csv(
                OUT_DIR / "tables" / f"shap_transformed_{analysis}_{outcome}_{model}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            base.to_csv(
                OUT_DIR / "tables" / f"shap_base_{analysis}_{outcome}_{model}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            all_base.append(base)
    summary = pd.concat(all_base, ignore_index=True)
    summary.to_csv(OUT_DIR / "shap_base_feature_importance_all_primary.csv", index=False, encoding="utf-8-sig")
    top10 = summary.groupby(["analysis", "outcome"], as_index=False).head(10)
    top10.to_csv(OUT_DIR / "shap_top10_base_features_all_primary.csv", index=False, encoding="utf-8-sig")
    print(OUT_DIR / "shap_base_feature_importance_all_primary.csv")


if __name__ == "__main__":
    main()
