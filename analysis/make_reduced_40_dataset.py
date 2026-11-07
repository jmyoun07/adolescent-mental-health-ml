from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

import run_protocol_ml as ml


PROJECT = ml.project_dir()
PROCESSED_DIR = os.path.join(PROJECT, "data", "processed")
REPORT_DIR = os.path.join(PROJECT, "reports")


def to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def recode_5_to_3(s: pd.Series, high_low_direction: str = "low_is_high") -> pd.Series:
    x = to_numeric(s)
    if high_low_direction == "low_is_high":
        return np.select([x.isin([1, 2]), x.eq(3), x.isin([4, 5])], [3, 2, 1], default=np.nan)
    return np.select([x.isin([1, 2]), x.eq(3), x.isin([4, 5])], [1, 2, 3], default=np.nan)


def recode_high_mid_low(s: pd.Series) -> pd.Series:
    # KYRBS self-rated academic/economic items commonly use 1=high ... 5=low.
    x = to_numeric(s)
    return np.select([x.isin([1, 2]), x.eq(3), x.isin([4, 5])], [3, 2, 1], default=np.nan)


def safe_mean(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)


def any_yes(frame: pd.DataFrame, columns: list[str], yes_values: set[float]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float")
    vals = frame[columns].apply(pd.to_numeric, errors="coerce")
    any_observed = vals.notna().any(axis=1)
    out.loc[any_observed] = vals.loc[any_observed].isin(yes_values).any(axis=1).astype(float)
    return out


def value_summary(s: pd.Series) -> str:
    nonmiss = s.dropna()
    if nonmiss.empty:
        return "전부 결측"
    unique = pd.Series(nonmiss.unique())
    try:
        values = sorted(pd.to_numeric(unique, errors="raise").tolist())
        if len(values) <= 15:
            return "관측값: " + ", ".join(format_value(v) for v in values)
        return f"범위 {format_value(min(values))}~{format_value(max(values))}; 고유값 {len(values)}개"
    except Exception:
        vals = sorted(map(str, unique.tolist()))[:15]
        suffix = "" if len(unique) <= 15 else f" 외 {len(unique) - 15}개"
        return "관측값: " + ", ".join(vals) + suffix


def format_value(v: Any) -> str:
    try:
        f = float(v)
        if f.is_integer():
            return str(int(f))
        return f"{f:.2f}".rstrip("0").rstrip(".")
    except Exception:
        return str(v)


def make_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = "" if pd.isna(row[col]) else str(row[col])
            vals.append(value.replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


FEATURE_SPECS = [
    # Sociodemographic / school / family
    ("SEX", "인구사회학", "성별", "원문항 유지", "성별은 대부분 KYRBS 정신건강 논문에서 기본 보정변수로 사용"),
    ("AGE", "인구사회학", "만 나이", "원문항 유지", "연령/학년 효과 보정"),
    ("GRADE", "인구사회학", "학년", "원문항 유지", "학년 효과 보정"),
    ("SCHOOL", "학교", "학교급", "원문항 유지", "중학교/일반계고/특성화계고 차이 보정"),
    ("STYPE", "학교", "학교유형", "원문항 유지", "학교유형 보정"),
    ("CTYPE", "지역", "도시규모", "원문항 유지", "시도 전체 대신 도시규모로 축약"),
    ("academic_achievement_3", "SES/학교", "학업성적 3수준", "E_S_RCRD: 상/중/하로 축약", "5점 척도를 상/중/하로 줄이는 문헌 관행 반영"),
    ("economic_status_3", "SES", "주관적 경제상태 3수준", "E_SES: 상/중/하로 축약", "5점 척도를 상/중/하로 줄이는 문헌 관행 반영"),
    ("E_RES", "가족/거주", "거주형태", "원문항 유지", "가족과 거주 여부 관련 보정"),
    ("living_with_both_parents", "가족/거주", "양부모 동거 여부", "E_FM_F_1, E_FM_M_3을 binary composite", "다수 가족구성 더미를 하나의 가족구조 변수로 축약"),
    ("parent_college_or_higher_any", "가족/SES", "부모 중 대졸 이상 여부", "E_EDU_F, E_EDU_M을 binary composite", "부모학력 2개 문항을 사회경제 proxy 하나로 축약"),
    # Mental health and sleep
    ("stress_3", "정신건강", "스트레스 3수준", "M_STR: 높음/보통/낮음", "문헌에서 high/some/low 또는 high vs others로 자주 축약"),
    ("sleep_satisfaction_3", "수면", "수면충족감 3수준", "M_SLP_EN: 좋음/보통/나쁨", "sleep quality/satisfaction 변수로 사용"),
    ("poor_sleep_satisfaction", "수면", "수면충족감 낮음 여부", "M_SLP_EN 4-5=1", "수면불충분 binary 변수로 문헌에서 자주 사용"),
    ("sleep_weekday_hours", "수면", "주중 수면시간", "취침/기상시각 파생", "수면시간 핵심 변수"),
    ("sleep_weekend_hours", "수면", "주말 수면시간", "취침/기상시각 파생", "주말 수면 보상 반영"),
    ("weekend_catchup_sleep_ratio", "수면", "주말/주중 수면비", "sleep_weekend_hours / sleep_weekday_hours", "weekend catch-up sleep 문헌 반영"),
    ("sleep_weekday_midpoint", "수면", "주중 수면 중간시각", "취침/기상시각 파생", "sleep timing 문헌 반영"),
    # Smartphone / digital phenotype
    ("smartphone_avg_daily_minutes", "디지털", "스마트폰 평균 일일 사용시간", "(주중*5+주말*2)/7", "스마트폰 사용시간 문헌의 기본 노출변수"),
    ("smartphone_time_group", "디지털", "스마트폰 사용시간 범주", "0, <2h, 2-4h, 4-6h, 6-8h, >8h", "시간 범주화 문헌 반영"),
    ("high_smartphone_4h", "디지털", "스마트폰 4시간 이상 여부", "평균 일일 사용시간 >=240분", "4시간 초과 threshold 문헌 반영"),
    ("smartphone_weekend_minus_weekday", "디지털", "주말-주중 스마트폰 사용시간 차이", "주말분-주중분", "주중/주말 사용패턴 차이 반영"),
    # Physical activity / diet / body
    ("PA_TOT", "신체활동", "하루 60분 이상 신체활동 일수", "원문항 유지", "신체활동 핵심 지표"),
    ("PA_MSC", "신체활동", "근력강화운동 일수", "원문항 유지", "신체활동 보조 지표"),
    ("sedentary_learning_avg", "좌식행동", "학습목적 평균 좌식시간", "PA_SWD_S, PA_SWK_S 가중평균", "주중/주말 좌식시간을 하나로 축약"),
    ("sedentary_nonlearning_avg", "좌식행동", "비학습목적 평균 좌식시간", "PA_SWD_N, PA_SWK_N 가중평균", "screen/sedentary behavior proxy"),
    ("F_BR", "식습관", "아침식사 빈도", "원문항 유지", "아침결식/식습관 문헌 반영"),
    ("F_FRUIT", "식습관", "과일 섭취 빈도", "원문항 유지", "건강식습관 지표"),
    ("F_FASTFOOD", "식습관", "패스트푸드 섭취 빈도", "원문항 유지", "불건강 식습관 지표"),
    ("F_WAT", "식습관", "물 섭취 빈도", "원문항 유지", "2020-2025 공통 식습관 변수"),
    ("BMI", "신체", "BMI", "HT, WT 기반 파생", "체중상태/신체이미지 보정"),
    ("PR_HT", "신체/주관건강", "주관적 건강인지", "원문항 유지", "정신건강 논문에서 자주 보정"),
    ("PR_BI", "신체이미지", "주관적 체형인지", "원문항 유지", "body image distortion 문헌 반영"),
    ("WC_MN", "신체/체중조절", "체중조절 노력", "원문항 유지", "체중조절/우울 문헌 반영"),
    # Substance / risk behavior
    ("current_alcohol", "음주", "현재 음주 여부", "AC_DAYS 기반 binary", "최근 30일 1일 이상 음주"),
    ("heavy_drunk_any", "음주", "최근 만취 경험 여부", "AC_DRUNK 기반 binary", "위험 음주 proxy"),
    ("any_current_tobacco", "흡연", "현재 담배제품 사용 여부", "일반담배/액상형/궐련형 통합", "흡연 제품을 하나의 current tobacco로 축약"),
    ("secondhand_smoke_exposure_any", "흡연노출", "간접흡연 노출 여부", "TC_SND_H, TC_SND_P 통합", "가정/공공장소 노출을 하나로 축약"),
    ("sexual_experience", "위험행동", "성경험 여부", "S_SI 기반 binary", "자살/우울 관련 위험행동 문헌 반영"),
    ("seatbelt_low_use_any", "위험행동", "안전벨트 낮은 착용 여부", "front/back/bus seatbelt 통합", "개별 안전벨트 문항 3개를 risk-taking proxy 하나로 축약"),
]


def build_reduced_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in ["YEAR", "W", "STRATA", "CLUSTER", "FPC"]:
        if col in df.columns:
            out[col] = df[col]

    # Targets
    for target in [
        "target_depressed_mood",
        "target_suicidal_ideation",
        "target_suicide_plan",
        "target_suicide_attempt_secondary",
    ]:
        if target in df.columns:
            out[target] = df[target]

    # Direct variables
    direct_cols = [
        "SEX",
        "AGE",
        "GRADE",
        "SCHOOL",
        "STYPE",
        "CTYPE",
        "E_RES",
        "sleep_weekday_hours",
        "sleep_weekend_hours",
        "sleep_weekday_midpoint",
        "smartphone_avg_daily_minutes",
        "high_smartphone_4h",
        "smartphone_weekend_minus_weekday",
        "PA_TOT",
        "PA_MSC",
        "F_BR",
        "F_FRUIT",
        "F_FASTFOOD",
        "F_WAT",
        "BMI",
        "PR_HT",
        "PR_BI",
        "WC_MN",
        "current_alcohol",
        "heavy_drunk_any",
        "any_current_tobacco",
        "sexual_experience",
    ]
    for col in direct_cols:
        if col in df.columns:
            out[col] = df[col]

    out["academic_achievement_3"] = recode_high_mid_low(df["E_S_RCRD"])
    out["economic_status_3"] = recode_high_mid_low(df["E_SES"])
    out["living_with_both_parents"] = (
        (to_numeric(df["E_FM_F_1"]).eq(1)) & (to_numeric(df["E_FM_M_3"]).eq(1))
    ).astype(float)
    parent_edu = df[["E_EDU_F", "E_EDU_M"]].apply(pd.to_numeric, errors="coerce")
    out["parent_college_or_higher_any"] = parent_edu.eq(3).any(axis=1).astype(float)
    out["stress_3"] = recode_5_to_3(df["M_STR"], high_low_direction="low_is_high")
    out["stress_high"] = to_numeric(df["M_STR"]).isin([1, 2]).astype(float)
    out["sleep_satisfaction_3"] = recode_5_to_3(df["M_SLP_EN"], high_low_direction="low_is_high")
    out["poor_sleep_satisfaction"] = to_numeric(df["M_SLP_EN"]).isin([4, 5]).astype(float)
    out["weekend_catchup_sleep_ratio"] = (
        to_numeric(df["sleep_weekend_hours"]) / to_numeric(df["sleep_weekday_hours"])
    ).replace([np.inf, -np.inf], np.nan)
    out["smartphone_time_group"] = pd.cut(
        to_numeric(df["smartphone_avg_daily_minutes"]),
        bins=[-0.1, 0, 120, 240, 360, 480, np.inf],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(float)
    out["sedentary_learning_avg"] = (
        to_numeric(df["PA_SWD_S"]) * 5 + to_numeric(df["PA_SWK_S"]) * 2
    ) / 7.0
    out["sedentary_nonlearning_avg"] = (
        to_numeric(df["PA_SWD_N"]) * 5 + to_numeric(df["PA_SWK_N"]) * 2
    ) / 7.0
    out["secondhand_smoke_exposure_any"] = any_yes(
        df, ["secondhand_smoke_home", "secondhand_smoke_public"], {1.0}
    )
    seatbelt = df[["front_seatbelt", "back_seatbelt", "bus_seatbelt"]].apply(pd.to_numeric, errors="coerce")
    out["seatbelt_low_use_any"] = np.where(seatbelt.notna().any(axis=1), seatbelt.ge(3).any(axis=1).astype(float), np.nan)

    feature_order = [spec[0] for spec in FEATURE_SPECS]
    ordered_cols = [
        c
        for c in [
            "YEAR",
            "W",
            "STRATA",
            "CLUSTER",
            "FPC",
            "target_depressed_mood",
            "target_suicidal_ideation",
            "target_suicide_plan",
            "target_suicide_attempt_secondary",
        ]
        if c in out.columns
    ] + feature_order
    return out.loc[:, ordered_cols]


def make_variable_table(reduced: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(reduced)
    for idx, (name, domain, content, coding, rationale) in enumerate(FEATURE_SPECS, start=1):
        missing_count = int(reduced[name].isna().sum())
        rows.append(
            {
                "no": idx,
                "variable_name": name,
                "domain": domain,
                "variable_content": content,
                "values": value_summary(reduced[name]),
                "missing_rate_pct": round(missing_count / n * 100, 2),
                "missing_count": missing_count,
                "coding_or_merge": coding,
                "selection_rationale": rationale,
                "main_analysis": "Y",
            }
        )
    return pd.DataFrame(rows)


def make_excluded_table() -> pd.DataFrame:
    rows = [
        ("CITY", "시도", "도시규모 CTYPE로 축약", "지역 고정효과가 필요할 때 민감도분석"),
        ("AGE_M", "월령", "AGE와 중복", "연령을 더 세밀하게 볼 때 사용"),
        ("A_FM, E_FM_*", "가족 구성원 더미", "living_with_both_parents로 축약", "가족구조 상세 분석 시 사용"),
        ("E_EDU_F, E_EDU_M", "부모학력 원문항", "parent_college_or_higher_any로 축약", "부모별 효과 분석 시 사용"),
        ("E_KRN_F, E_KRN_M", "부모 한국 국적", "본 연구 핵심 종속변수와 직접성 낮음", "다문화 subgroup 연구 시 사용"),
        ("M_STR 원점수", "스트레스 5점", "stress_3, stress_high로 축약", "비선형 효과 확인 시 사용"),
        ("M_SLP_EN 원점수", "수면충족감 5점", "sleep_satisfaction_3, poor_sleep_satisfaction로 축약", "비선형 효과 확인 시 사용"),
        ("PA_SWD_S, PA_SWK_S", "학습목적 좌식시간 주중/주말", "sedentary_learning_avg로 통합", "주중/주말 별도 효과 확인 시 사용"),
        ("PA_SWD_N, PA_SWK_N", "비학습목적 좌식시간 주중/주말", "sedentary_nonlearning_avg로 통합", "screen sedentary 분석 시 사용"),
        ("INT_SPWD_TM, INT_SPWK_TM", "스마트폰 주중/주말 시간", "평균, 범주, 주말-주중 차이로 통합", "사용패턴 세부 분석 시 사용"),
        ("current_cigarette, current_ecig, current_htp", "담배제품별 현재 사용", "any_current_tobacco로 통합", "담배제품별 효과 분석 시 사용"),
        ("TC_SND_H, TC_SND_P", "간접흡연 장소별 노출", "secondhand_smoke_exposure_any로 통합", "장소별 노출 분석 시 사용"),
        ("front/back/bus_seatbelt", "안전벨트 3문항", "seatbelt_low_use_any로 통합", "주분석 제외 민감도분석 가능"),
        ("lunch_toothbrushing", "점심 후 칫솔질", "제외", "종속변수와 이론적 직접성 낮음"),
        ("contraception_use", "피임 실천", "제외", "성경험자 조건부 문항, 전체 모델 결측 과다"),
        ("alcohol_days_level", "음주일수 강도", "제외", "current_alcohol/heavy_drunk_any로 대체, 조건부 결측 큼"),
        ("cigarette_days_level", "흡연일수 강도", "제외", "any_current_tobacco로 대체, 조건부 결측 큼"),
        ("AC_ACCESS, TC_ACCESS", "주류/담배 구매 용이성", "제외", "조건부/해석 중복 가능"),
        ("DR_HAB_PUR", "약물사용 경험", "pooled 주분석 제외", "2020-2021 미조사, 단일연도/2022+ 모델에서 사용"),
        ("F_CAFF_A, F_SWD_A, F_ZERO, F_VEG_A, F_MD, F_NE, PA_VIG_D", "일부 연도 식습관/신체활동", "pooled 주분석 제외", "연도별 미조사, 단일연도 모델 후보"),
    ]
    return pd.DataFrame(rows, columns=["excluded_or_merged_variables", "content", "decision", "note"])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    df, _ = ml.build_analysis_dataset(PROJECT, force=False)
    reduced = build_reduced_dataset(df)
    variable_table = make_variable_table(reduced)
    excluded = make_excluded_table()

    dataset_path = os.path.join(PROCESSED_DIR, "kyrbs_2020_2025_reduced_40_features.csv")
    table_path = os.path.join(REPORT_DIR, "protocol_reduced_40_variable_table.csv")
    excluded_path = os.path.join(REPORT_DIR, "protocol_reduced_40_excluded_or_merged_variables.csv")
    md_path = os.path.join(REPORT_DIR, "protocol_reduced_40_variable_table.md")
    metadata_path = os.path.join(REPORT_DIR, "protocol_reduced_40_metadata.json")

    reduced.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    variable_table.to_csv(table_path, index=False, encoding="utf-8-sig")
    excluded.to_csv(excluded_path, index=False, encoding="utf-8-sig")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Reduced 40-feature variable table\n\n")
        f.write(make_markdown(variable_table))
        f.write("\n\n# Excluded or merged variables\n\n")
        f.write(make_markdown(excluded))
        f.write("\n")

    metadata = {
        "n_rows": int(len(reduced)),
        "n_predictor_features": len(FEATURE_SPECS),
        "feature_columns": [spec[0] for spec in FEATURE_SPECS],
        "target_columns": [c for c in reduced.columns if c.startswith("target_")],
        "design_columns": [c for c in ["YEAR", "W", "STRATA", "CLUSTER", "FPC"] if c in reduced.columns],
        "removed_from_main_analysis": [
            "lunch_toothbrushing",
            "front_seatbelt",
            "back_seatbelt",
            "bus_seatbelt",
            "contraception_use",
            "alcohol_days_level",
            "cigarette_days_level",
        ],
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(dataset_path)
    print(table_path)
    print(excluded_path)
    print(md_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
