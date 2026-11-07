from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports"
RAW_DIR = PROJECT / "data" / "raw_sav"
PROCESSED = PROJECT / "data" / "processed" / "kyrbs_2020_2025_protocol_analysis_dataset.csv"
SOURCE = REPORTS / "pubmed_lit_reduced_variable_decisions.csv"
OUTPUT = REPORTS / "pubmed_lit_reduced_variable_decisions_40plus.csv"
BACKUP = REPORTS / "pubmed_lit_reduced_variable_decisions_before_40plus.csv"


def binary_yes2(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return pd.Series(np.where(s == 2, 1.0, np.where(s == 1, 0.0, np.nan)), index=s.index)


def missing_summary(s: pd.Series, total_n: int | None = None) -> str:
    if total_n is None:
        total_n = len(s)
    n_missing = int(pd.isna(s).sum())
    return f"{n_missing / total_n * 100:.2f}% ({n_missing})"


def year_availability(varnames: list[str]) -> str:
    years = []
    for sav in sorted(RAW_DIR.glob("kyrbs202*.sav")):
        year = sav.stem[-4:]
        meta = pyreadstat.read_sav(str(sav), metadataonly=True)[1]
        if all(v in meta.column_names for v in varnames):
            years.append(year)
    return f"{years[0]}-{years[-1]}" if years and len(years) > 1 else (years[0] if years else "not available")


def load_raw_candidates() -> pd.DataFrame:
    needed = [
        "I_SCH_TRT",
        "V_TRT",
        "AS_DG_LT",
        "RH_DG_LT",
        "ECZ_DG_LT",
        "M_LON",
        *[f"M_GAD_{i}" for i in range(1, 8)],
    ]
    frames = []
    for sav in sorted(RAW_DIR.glob("kyrbs202*.sav")):
        year = int(sav.stem[-4:])
        meta = pyreadstat.read_sav(str(sav), metadataonly=True)[1]
        usecols = [c for c in needed if c in meta.column_names]
        if usecols:
            frame, _ = pyreadstat.read_sav(str(sav), usecols=usecols, apply_value_formats=False)
        else:
            frame = pd.DataFrame(index=range(meta.number_rows))
        for col in needed:
            if col not in frame.columns:
                frame[col] = np.nan
        frame["YEAR"] = year
        frames.append(frame[["YEAR", *needed]])
    return pd.concat(frames, ignore_index=True).replace([7777, 8888, 9999], np.nan)


def build_added_rows(total_n: int) -> list[dict[str, str]]:
    raw = load_raw_candidates()
    proc_cols = [
        "AC_LT",
        "TC_LT",
        "TC_EC_LT",
        "TC_HTP_LT",
        "AC_ACCESS",
        "TC_ACCESS",
    ]
    proc = pd.read_csv(PROCESSED, usecols=proc_cols, low_memory=False).replace([7777, 8888, 9999], np.nan)

    gad_cols = [f"M_GAD_{i}" for i in range(1, 8)]
    gad7_total = (raw[gad_cols] - 1).sum(axis=1, min_count=7)
    loneliness_3 = pd.Series(np.nan, index=raw.index)
    loneliness_3[raw["M_LON"].isin([1, 2])] = 1
    loneliness_3[raw["M_LON"].eq(3)] = 2
    loneliness_3[raw["M_LON"].isin([4, 5])] = 3

    disease_cols = {
        "asthma_lifetime": binary_yes2(raw["AS_DG_LT"]),
        "allergic_rhinitis_lifetime": binary_yes2(raw["RH_DG_LT"]),
        "atopic_dermatitis_lifetime": binary_yes2(raw["ECZ_DG_LT"]),
    }
    disease_frame = pd.DataFrame(disease_cols)
    atopic_any = disease_frame.max(axis=1, skipna=True)
    atopic_count = disease_frame.sum(axis=1, min_count=1)

    tobacco_components = pd.DataFrame(
        {
            "lifetime_cigarette_experience": binary_yes2(proc["TC_LT"]),
            "lifetime_ecig_experience": binary_yes2(proc["TC_EC_LT"]),
            "lifetime_htp_experience": binary_yes2(proc["TC_HTP_LT"]),
        }
    )
    lifetime_tobacco_any = tobacco_components.max(axis=1, skipna=True)

    # Access variables are conditional; skipped/not-applicable responses are coded as no exposure
    # for a compact sensitivity proxy rather than a core predictor.
    alcohol_access_easy = pd.Series(np.where(proc["AC_ACCESS"].notna(), 1.0, 0.0), index=proc.index)
    tobacco_access_easy = pd.Series(np.where(proc["TC_ACCESS"].notna(), 1.0, 0.0), index=proc.index)
    substance_access_any = pd.concat([alcohol_access_easy, tobacco_access_easy], axis=1).max(axis=1)

    additions = [
        (
            "main_plus_keep",
            "school_injury_treatment_any",
            "최근 12개월 학교 손상 병원치료 경험",
            "I_SCH_TRT: 2=있음, 1=없음으로 binary",
            "2020-2025 main-plus",
            binary_yes2(raw["I_SCH_TRT"]),
            "2020-2025",
            "프로토콜의 bullying/violence/injury 축 및 KYRBS suicide ML 문헌의 school injury 변수 반영",
        ),
        (
            "main_plus_keep",
            "lifetime_alcohol_experience",
            "평생 음주 경험",
            "AC_LT: 2=있음, 1=없음으로 binary",
            "2020-2025 main-plus",
            binary_yes2(proc["AC_LT"]),
            "2020-2025",
            "현재 음주만으로 놓치는 위험행동 이력 보완",
        ),
        (
            "main_plus_keep",
            "lifetime_tobacco_any",
            "평생 담배제품 경험 통합",
            "TC_LT, TC_EC_LT, TC_HTP_LT 중 하나라도 경험=1",
            "2020-2025 main-plus",
            lifetime_tobacco_any,
            "2020-2025",
            "프로토콜의 tobacco/vaping 축 반영, 현재 흡연과 다른 lifetime risk proxy",
        ),
        (
            "sensitivity_keep",
            "lifetime_cigarette_experience",
            "평생 일반담배 경험",
            "TC_LT: 2=있음, 1=없음으로 binary",
            "tobacco decomposition sensitivity",
            tobacco_components["lifetime_cigarette_experience"],
            "2020-2025",
            "통합 담배제품 변수의 구성요소 확인용",
        ),
        (
            "enhanced_keep",
            "violence_treatment_any",
            "폭력으로 인한 병원치료 경험",
            "V_TRT: 2 이상=있음, 1=없음",
            "2020-2024 enhanced",
            pd.Series(np.where(raw["V_TRT"].notna(), (raw["V_TRT"] > 1).astype(float), np.nan), index=raw.index),
            "2020-2024",
            "프로토콜의 bullying victimization 축, 2019 KYRBS suicide ML 문헌의 violence 변수 반영",
        ),
        (
            "enhanced_keep",
            "gad7_total",
            "범불안장애 GAD-7 총점",
            "M_GAD_1~7을 0-3점으로 변환 후 합산(0-21)",
            "2020-2024 enhanced",
            gad7_total,
            "2020-2024",
            "최신 KYRBS 자살위험 ML 문헌에서 불안은 핵심 정신건강 predictor",
        ),
        (
            "enhanced_keep",
            "gad7_high10",
            "범불안장애 고위험",
            "gad7_total >= 10",
            "2020-2024 enhanced",
            pd.Series(np.where(gad7_total.notna(), (gad7_total >= 10).astype(float), np.nan), index=raw.index),
            "2020-2024",
            "GAD-7 표준 threshold 기반 민감도/해석용",
        ),
        (
            "enhanced_keep",
            "loneliness_3",
            "외로움 3수준",
            "M_LON: 낮음/보통/높음 3수준",
            "2020-2024 enhanced",
            loneliness_3,
            "2020-2024",
            "최소위험요인 논문에서 외로움은 자살생각/시도 핵심 predictor",
        ),
        (
            "enhanced_keep",
            "loneliness_high",
            "외로움 높음 여부",
            "M_LON 4-5=1",
            "2020-2024 enhanced",
            pd.Series(np.where(raw["M_LON"].notna(), raw["M_LON"].isin([4, 5]).astype(float), np.nan), index=raw.index),
            "2020-2024",
            "외로움 threshold 기반 민감도/해석용",
        ),
        (
            "enhanced_keep",
            "asthma_lifetime",
            "평생 천식 의사진단",
            "AS_DG_LT: 2=있음, 1=없음으로 binary",
            "2020-2024 enhanced",
            disease_frame["asthma_lifetime"],
            "2020-2024",
            "KYRBS suicide attempt ML 문헌의 comorbidity 변수",
        ),
        (
            "enhanced_keep",
            "allergic_rhinitis_lifetime",
            "평생 알레르기비염 의사진단",
            "RH_DG_LT: 2=있음, 1=없음으로 binary",
            "2020-2024 enhanced",
            disease_frame["allergic_rhinitis_lifetime"],
            "2020-2024",
            "JMIR allergic rhinitis suicidality 논문 및 KYRBS comorbidity 축 반영",
        ),
        (
            "enhanced_keep",
            "atopic_dermatitis_lifetime",
            "평생 아토피피부염 의사진단",
            "ECZ_DG_LT: 2=있음, 1=없음으로 binary",
            "2020-2024 enhanced",
            disease_frame["atopic_dermatitis_lifetime"],
            "2020-2024",
            "KYRBS suicide attempt ML 문헌의 comorbidity 변수",
        ),
        (
            "enhanced_keep",
            "atopic_disease_any",
            "아토피성 질환 경험 여부",
            "천식/알레르기비염/아토피피부염 중 하나라도 있음=1",
            "2020-2024 enhanced",
            atopic_any,
            "2020-2024",
            "질환 3개 문항을 하나의 comorbidity proxy로 축약",
        ),
        (
            "sensitivity_keep",
            "atopic_disease_count",
            "아토피성 질환 개수",
            "천식/알레르기비염/아토피피부염 binary 합계(0-3)",
            "2020-2024 sensitivity",
            atopic_count,
            "2020-2024",
            "질환 부담 정도를 반영하는 보조 변수",
        ),
        (
            "sensitivity_keep",
            "substance_access_reported_any",
            "주류/담배 구매 접근성 응답 경험",
            "AC_ACCESS 또는 TC_ACCESS 응답 있음=1, 비해당/무응답=0",
            "2020-2025 sensitivity",
            substance_access_any,
            "2020-2025",
            "접근성 변수는 조건부 문항이므로 주분석보다 보조분석 후보",
        ),
    ]

    rows: list[dict[str, str]] = []
    for decision, name, content, coding, scope, series, years, reason in additions:
        rows.append(
            {
                "decision": decision,
                "variable_name": name,
                "content": content,
                "coding_or_merge": coding,
                "use_scope": scope,
                "missing_rate_2020_2025": missing_summary(series, total_n),
                "year_availability": years,
                "main_reason": reason,
            }
        )
    return rows


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(SOURCE, encoding="utf-8-sig")
    if not BACKUP.exists():
        shutil.copy2(SOURCE, BACKUP)

    filtered = original[~original["decision"].astype(str).str.lower().str.startswith("drop")].copy()
    merged_out = {
        "lifetime_ecig_experience",
        "lifetime_htp_experience",
        "lifetime_electronic_cigarette_any",
    }
    filtered = filtered[~filtered["variable_name"].isin(merged_out)].copy()
    total_n = sum(1 for _ in pd.read_csv(PROCESSED, usecols=["YEAR"], chunksize=100_000))
    # The chunk count above gives chunks, not rows; get N from the target column cheaply.
    total_n = len(pd.read_csv(PROCESSED, usecols=["YEAR"], low_memory=False))
    additions = pd.DataFrame(build_added_rows(total_n))

    existing_names = set(filtered["variable_name"].astype(str))
    additions = additions[~additions["variable_name"].isin(existing_names)]
    final = pd.concat([filtered, additions], ignore_index=True)

    final.to_csv(SOURCE, index=False, encoding="utf-8-sig")
    final.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print(f"rows_before={len(original)}")
    print(f"rows_after_drop_removed={len(filtered)}")
    print(f"added_rows={len(additions)}")
    print(f"final_rows={len(final)}")
    print(f"updated={SOURCE}")
    print(f"copy={OUTPUT}")
    print(f"backup={BACKUP}")
    print(final.groupby('decision').size().to_string())


if __name__ == "__main__":
    main()
