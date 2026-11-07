from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT / "reports"
OUT_CSV = REPORT_DIR / "revised_yrbss_external_summary_2026_07_09.csv"
OUT_MD = REPORT_DIR / "revised_yrbss_external_summary_2026_07_09.md"

SCENARIOS = {
    "primary": PROJECT / "outputs" / "external_validation_yrbss_2023_revised_shared_primary" / "primary_model_by_kyrbs_validation.csv",
    "no_digital": PROJECT / "outputs" / "external_validation_yrbss_2023_revised_shared_no_digital" / "primary_model_by_kyrbs_validation.csv",
    "no_depressed_predictor": PROJECT / "outputs" / "external_validation_yrbss_2023_revised_shared_no_depressed_predictor" / "primary_model_by_kyrbs_validation.csv",
}


def fmt(value, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    rows = []
    for scenario, path in SCENARIOS.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append(
                {
                    "scenario": scenario,
                    "outcome": r["outcome"],
                    "model": r["model"],
                    "features_n": r["features_n"],
                    "validation_auprc": r["validation_auprc"],
                    "internal_2024_auroc": r["internal_2024_auroc"],
                    "internal_2024_auprc": r["internal_2024_auprc"],
                    "external_yrbss_auroc": r["external_yrbss_auroc"],
                    "external_yrbss_auprc": r["external_yrbss_auprc"],
                    "external_yrbss_brier": r["external_yrbss_brier"],
                    "external_yrbss_calibration_intercept": r["external_yrbss_calibration_intercept"],
                    "external_yrbss_calibration_slope": r["external_yrbss_calibration_slope"],
                    "external_yrbss_weighted_auroc": r.get("external_yrbss_weighted_weighted_auroc"),
                    "external_yrbss_weighted_auprc": r.get("external_yrbss_weighted_weighted_auprc"),
                }
            )
    out = pd.DataFrame(rows)
    primary = out[out["scenario"] == "primary"][
        ["outcome", "external_yrbss_auroc", "external_yrbss_auprc", "internal_2024_auroc", "internal_2024_auprc"]
    ].rename(
        columns={
            "external_yrbss_auroc": "primary_external_auroc",
            "external_yrbss_auprc": "primary_external_auprc",
            "internal_2024_auroc": "primary_internal_auroc",
            "internal_2024_auprc": "primary_internal_auprc",
        }
    )
    out = out.merge(primary, on="outcome", how="left")
    out["delta_external_auroc_vs_primary"] = out["external_yrbss_auroc"] - out["primary_external_auroc"]
    out["delta_external_auprc_vs_primary"] = out["external_yrbss_auprc"] - out["primary_external_auprc"]
    out["delta_internal_auroc_vs_primary"] = out["internal_2024_auroc"] - out["primary_internal_auroc"]
    out["delta_internal_auprc_vs_primary"] = out["internal_2024_auprc"] - out["primary_internal_auprc"]
    out = out.sort_values(["outcome", "scenario"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    display = out.copy()
    for col in display.columns:
        if col not in {"scenario", "outcome", "model"}:
            display[col] = display[col].map(fmt)
    key_cols = [
        "outcome",
        "scenario",
        "model",
        "features_n",
        "internal_2024_auroc",
        "external_yrbss_auroc",
        "external_yrbss_auprc",
        "delta_external_auroc_vs_primary",
        "delta_external_auprc_vs_primary",
        "external_yrbss_calibration_intercept",
        "external_yrbss_calibration_slope",
    ]
    lines = [
        "# Revised YRBSS External Summary",
        "",
        "작성일: 2026-07-09",
        "",
        "이 문서는 revised shared-feature model의 YRBSS 2023 transportability 평가와 sensitivity analysis를 합친 것이다.",
        "",
        "## Scenario Definitions",
        "",
        "- `primary`: YRBSS와 조화 가능한 revised shared-feature model.",
        "- `no_digital`: `digital_high_use` 제거.",
        "- `no_depressed_predictor`: suicide outcomes에서 `depressed_mood_for_suicide_models` 제거.",
        "",
        "## Results",
        "",
        markdown_table(display[key_cols], key_cols),
        "",
        "## Notes",
        "",
        "- This is not external validation of the full KYRBS revised model; it is a shared-feature transportability evaluation.",
        "- `digital_high_use` remains a weak proxy: KYRBS smartphone >=4h/day versus YRBSS social media several times/day.",
        "- GAD-7, loneliness, stress, poor sleep recovery, weekday/weekend smartphone minutes, and high smartphone 6h are unavailable in YRBSS 2023 national data.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_CSV)
    print(OUT_MD)


if __name__ == "__main__":
    main()
