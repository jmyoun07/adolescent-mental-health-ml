from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadstat


PROJECT = Path(__file__).resolve().parents[1]
KEYWORDS = ["WT", "WGT", "WEIGHT", "STRAT", "CLUSTER", "PSU", "KSTRATA", "SAMPLE"]


def hits(cols: list[str]) -> list[str]:
    return [c for c in cols if any(k in c.upper() for k in KEYWORDS)]


def main() -> None:
    rows = []
    for path in [
        PROJECT / "data" / "processed" / "kyrbs_2020_2025_protocol_analysis_dataset.csv",
        PROJECT / "data" / "processed" / "kyrbs_2020_2025_lit_reduced_revised_dataset.csv",
    ]:
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        rows.append({"source": str(path), "survey_like_columns": ";".join(hits(cols))})
    for path in sorted((PROJECT / "data" / "raw_sav").glob("kyrbs202*.sav")):
        _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
        rows.append({"source": str(path), "survey_like_columns": ";".join(hits(meta.column_names))})
    out = pd.DataFrame(rows)
    out_path = PROJECT / "outputs" / "remaining_methodology_fixes_2026_07_09" / "kyrbs_survey_design_variable_scan.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out_path)


if __name__ == "__main__":
    main()
