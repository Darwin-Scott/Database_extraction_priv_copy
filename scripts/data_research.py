from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from dbx.paths import OUT, RAW_DATA


DIFF_PAIRS = [
    ("headline", "original_headline"),
    ("current_company", "original_current_company"),
    ("current_company_position", "original_current_company_position"),
]

EXAMPLE_COLUMNS = [
    "mini_profile_actual_at",
    "location_name",
    "industry",
    "industry_actual_at",
    "badges_job_seeker",
    "badges_open_link",
    "current_company_position",
    "original_current_company_position",
    "education_1",
    "education_degree_1",
    "education_fos_1",
    "education_description_1",
    "education_2",
    "education_degree_2",
    "education_fos_2",
    "education_description_2",
    "education_3",
    "education_degree_3",
    "education_fos_3",
    "education_description_3",
    "languages",
    "skills",
    "phone_1",
    "phone_type_1",
    "phone_2",
    "phone_type_2",
    "messenger_1",
    "messenger_provider_1",
    "messenger_2",
    "messenger_provider_2",
    "tags",
    "note",
]

SPARSITY_THRESHOLD = 0.80


def detect_delimiter(csv_path: Path, sample_bytes: int = 20000) -> str:
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(sample_bytes)
    dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
    return dialect.delimiter


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return value


def load_raw_csvs(raw_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        delimiter = detect_delimiter(csv_path)
        frame = pd.read_csv(
            csv_path,
            sep=delimiter,
            engine="python",
            encoding="utf-8",
            encoding_errors="replace",
            on_bad_lines="skip",
            dtype=str,
        )
        frame = frame.copy()
        frame.insert(0, "source_file", csv_path.name)
        frame.insert(1, "source_row", range(2, len(frame) + 2))
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.where(pd.notna(combined), None)
    return combined


def normalized_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index, dtype="object")
    return frame[column].map(clean_value)


def build_diff_frame(frame: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    left_values = normalized_series(frame, left)
    right_values = normalized_series(frame, right)
    mismatch_mask = left_values.ne(right_values) & ~(left_values.isna() & right_values.isna())

    diff = pd.DataFrame(
        {
            "source_file": frame["source_file"],
            "source_row": frame["source_row"],
            left: left_values,
            right: right_values,
        }
    )
    diff = diff.loc[mismatch_mask].copy()
    return diff.reset_index(drop=True)


def build_examples_frame(frame: pd.DataFrame, columns: list[str], limit: int = 3) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        if column not in frame.columns:
            rows.append(
                {
                    "column": column,
                    "example_index": None,
                    "source_file": None,
                    "source_row": None,
                    "value": None,
                    "status": "missing_column",
                }
            )
            continue

        values = normalized_series(frame, column)
        example_count = 0
        seen: set[str] = set()

        for idx, value in values.items():
            if value is None:
                continue

            key = str(value)
            if key in seen:
                continue

            seen.add(key)
            example_count += 1
            rows.append(
                {
                    "column": column,
                    "example_index": example_count,
                    "source_file": frame.at[idx, "source_file"],
                    "source_row": frame.at[idx, "source_row"],
                    "value": value,
                    "status": "ok",
                }
            )
            if example_count >= limit:
                break

        if example_count == 0:
            rows.append(
                {
                    "column": column,
                    "example_index": None,
                    "source_file": None,
                    "source_row": None,
                    "value": None,
                    "status": "no_non_empty_values",
                }
            )

    return pd.DataFrame(rows)


def build_sparsity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(frame)
    rows: list[dict[str, object]] = []

    for column in frame.columns:
        if column in {"source_file", "source_row"}:
            continue

        values = normalized_series(frame, column)
        non_empty = int(values.notna().sum())
        empty = total_rows - non_empty
        rows.append(
            {
                "column": column,
                "non_empty_count": non_empty,
                "empty_count": empty,
                "non_empty_pct": round(non_empty / total_rows, 4) if total_rows else 0.0,
                "empty_pct": round(empty / total_rows, 4) if total_rows else 0.0,
            }
        )

    sparsity = pd.DataFrame(rows)
    sparsity = sparsity.sort_values(["empty_pct", "column"], ascending=[False, True]).reset_index(drop=True)
    return sparsity


def write_summary(
    summary_path: Path,
    frame: pd.DataFrame,
    diff_results: dict[tuple[str, str], pd.DataFrame],
    examples: pd.DataFrame,
    sparse_columns: pd.DataFrame,
) -> None:
    csv_files = sorted(frame["source_file"].dropna().unique().tolist())
    lines = [
        "# Data Research Summary",
        "",
        f"- Rows scanned: {len(frame)}",
        f"- CSV files scanned: {len(csv_files)}",
        f"- Files: {', '.join(csv_files)}",
        f"- Sparsity threshold: empty_pct >= {SPARSITY_THRESHOLD:.0%}",
        "",
        "## Column Differences",
    ]

    for left, right in DIFF_PAIRS:
        diff = diff_results[(left, right)]
        lines.append(f"- {left} vs {right}: {len(diff)} differing rows")

    lines.append("")
    lines.append("## Sparse Columns")
    top_sparse = sparse_columns.loc[sparse_columns["empty_pct"] >= SPARSITY_THRESHOLD].head(40)
    if top_sparse.empty:
        lines.append("- No columns crossed the configured threshold.")
    else:
        for _, row in top_sparse.iterrows():
            lines.append(
                f"- {row['column']}: empty {row['empty_count']}/{len(frame)} "
                f"({row['empty_pct']:.1%}), non-empty {row['non_empty_count']}"
            )

    lines.append("")
    lines.append("## Example Values")
    for column in EXAMPLE_COLUMNS:
        lines.append(f"- {column}:")
        column_rows = examples.loc[examples["column"] == column]
        for _, row in column_rows.iterrows():
            if row["status"] != "ok":
                lines.append(f"  - {row['status']}")
                continue
            preview = str(row["value"]).replace("\n", " ").strip()
            if len(preview) > 180:
                preview = preview[:177].rstrip() + "..."
            lines.append(
                f"  - [{row['source_file']}:{int(row['source_row'])}] {preview}"
            )

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = OUT / "data_research"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_raw_csvs(RAW_DATA)

    diff_results: dict[tuple[str, str], pd.DataFrame] = {}
    for left, right in DIFF_PAIRS:
        diff = build_diff_frame(frame, left, right)
        diff_results[(left, right)] = diff
        diff.to_csv(out_dir / f"diff__{left}__vs__{right}.csv", index=False, encoding="utf-8")

    examples = build_examples_frame(frame, EXAMPLE_COLUMNS, limit=3)
    examples.to_csv(out_dir / "column_examples.csv", index=False, encoding="utf-8")

    sparsity = build_sparsity_frame(frame)
    sparsity.to_csv(out_dir / "column_sparsity.csv", index=False, encoding="utf-8")
    sparse_only = sparsity.loc[sparsity["empty_pct"] >= SPARSITY_THRESHOLD].copy()
    sparse_only.to_csv(out_dir / "column_sparsity_sparse_only.csv", index=False, encoding="utf-8")

    summary_path = out_dir / "summary.md"
    write_summary(summary_path, frame, diff_results, examples, sparsity)

    print(f"Rows scanned: {len(frame)}")
    print(f"Output directory: {out_dir}")
    for left, right in DIFF_PAIRS:
        print(f"{left} vs {right}: {len(diff_results[(left, right)])} differing rows")
    print(f"Examples file: {out_dir / 'column_examples.csv'}")
    print(f"Sparsity file: {out_dir / 'column_sparsity.csv'}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
