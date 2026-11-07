from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


VAR_LABEL_RE = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+\"(?P<label>.*)\"\.?\s*$")
VALUE_RE = re.compile(r"^\s*(?P<code>[-+]?\d+(?:\.\d+)?)\s+\"(?P<label>.*)\"\s*$")


def block_between(text: str, start: str, end: str) -> str:
    pattern = re.compile(start + r"(?P<body>.*?)" + end, flags=re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if not match:
        return ""
    return match.group("body")


def parse_variable_labels(spss_text: str) -> dict[str, str]:
    block = block_between(spss_text, r"VARIABLE\s+LABELS", r"VALUE\s+LABELS")
    labels: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.endswith("."):
            line = line[:-1]
        match = VAR_LABEL_RE.match(line)
        if match:
            labels[match.group("name")] = match.group("label")
    return labels


def parse_value_labels(spss_text: str) -> dict[str, dict[str, str]]:
    block = block_between(spss_text, r"VALUE\s+LABELS", r"MISSING\s+VALUES")
    value_labels: dict[str, dict[str, str]] = {}
    current: list[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in {"/", "/."}:
            current = []
            continue
        value_match = VALUE_RE.match(line)
        if value_match and current:
            for var in current:
                value_labels.setdefault(var, {})[value_match.group("code")] = value_match.group("label")
            continue
        if '"' not in line:
            line = line.rstrip(".")
            current = [token for token in re.split(r"\s+", line) if token]
            for var in current:
                value_labels.setdefault(var, {})
    return value_labels


def summarize_values(raw: pd.DataFrame, name: str) -> tuple[int, int, float, str]:
    s = raw[name].astype(str) if name in raw.columns else pd.Series(dtype=str)
    missing = s.isin(["", ".", "nan", "NaN"])
    nonmissing_n = int((~missing).sum())
    missing_n = int(missing.sum())
    missing_rate = float(missing.mean()) if len(s) else 1.0
    top = s[~missing].value_counts().head(12)
    top_values = "; ".join(f"{idx}:{count}" for idx, count in top.items())
    return nonmissing_n, missing_n, missing_rate, top_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--columns", type=Path, required=True)
    parser.add_argument("--spss", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    columns = pd.read_csv(args.columns)
    spss_text = args.spss.read_text(encoding="utf-8", errors="replace")
    variable_labels = parse_variable_labels(spss_text)
    value_labels = parse_value_labels(spss_text)
    raw = pd.read_csv(args.raw_csv, dtype=str, keep_default_na=False, low_memory=False)

    rows = []
    for row in columns.to_dict("records"):
        name = str(row["name"])
        labels = value_labels.get(name, {})
        value_label_text = "; ".join(f"{code}={label}" for code, label in labels.items())
        nonmissing_n, missing_n, missing_rate, top_values = summarize_values(raw, name)
        rows.append(
            {
                "variable_name": name,
                "start": row["start"],
                "end": row["end"],
                "width": row["width"],
                "storage_type": row["type"],
                "variable_label": variable_labels.get(name, ""),
                "value_labels": value_label_text,
                "nonmissing_n": nonmissing_n,
                "missing_n": missing_n,
                "missing_rate": missing_rate,
                "top_observed_values": top_values,
            }
        )
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"rows={len(out)}")
    print(args.out)


if __name__ == "__main__":
    main()
