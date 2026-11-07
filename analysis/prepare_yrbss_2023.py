from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


DATA_LIST_RE = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<start>\d+)-(?P<end>\d+)(?:\s+\((?P<type>A)\))?")


def parse_spss_data_list(spss_path: Path) -> list[dict[str, object]]:
    text = spss_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"DATA\s+LIST\s+FILE=.*?/\s*(?P<body>.*?)\.\s", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not find DATA LIST block in {spss_path}")

    columns: list[dict[str, object]] = []
    for item in DATA_LIST_RE.finditer(match.group("body")):
        start = int(item.group("start"))
        end = int(item.group("end"))
        columns.append(
            {
                "name": item.group("name"),
                "start": start,
                "end": end,
                "width": end - start + 1,
                "type": "string" if item.group("type") == "A" else "numeric_or_code",
            }
        )
    if not columns:
        raise ValueError(f"No columns parsed from DATA LIST block in {spss_path}")
    return columns


def convert(dat_path: Path, spss_path: Path, out_csv: Path, out_columns: Path) -> None:
    columns = parse_spss_data_list(spss_path)
    colspecs = [(int(col["start"]) - 1, int(col["end"])) for col in columns]
    names = [str(col["name"]) for col in columns]

    df = pd.read_fwf(dat_path, colspecs=colspecs, names=names, dtype=str, keep_default_na=False)
    df = df.apply(lambda s: s.str.strip())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with out_columns.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "start", "end", "width", "type"])
        writer.writeheader()
        writer.writerows(columns)

    print(f"rows={len(df)}")
    print(f"columns={len(df.columns)}")
    print(f"csv={out_csv}")
    print(f"columns_csv={out_columns}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CDC 2023 National YRBS fixed-width data to CSV.")
    parser.add_argument("--dat", type=Path, required=True)
    parser.add_argument("--spss", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-columns", type=Path, required=True)
    args = parser.parse_args()
    convert(args.dat, args.spss, args.out_csv, args.out_columns)


if __name__ == "__main__":
    main()
