import csv
import pandas as pd
from pathlib import Path
from dbx.paths import RAW_DATA

def detect_delimiter(csv_path, sample_bytes=20000):
    """Detect delimiter using csv.Sniffer on a sample."""
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(sample_bytes)
    dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
    return dialect.delimiter

def extract_schema_overview(csv_path, output_path="schema_overview.txt"):
    csv_path = Path(csv_path)

    # 1) detect delimiter
    delim = detect_delimiter(csv_path)

    # 2) read CSV robustly
    df = pd.read_csv(
        csv_path,
        sep=delim,
        engine="python",          # more tolerant than default C engine
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="skip"      # skips malformed rows instead of crashing
    )

    lines = []
    for col in df.columns:
        s = df[col]

        # first non-null example
        example = s.dropna().iloc[0] if not s.dropna().empty else "NULL"

        # shorten long text
        if isinstance(example, str) and len(example) > 120:
            example = example[:120] + "..."

        lines.append(f"{col}: {example} ({s.dtype})")

    out_path = csv_path.parent / output_path
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Detected delimiter: {repr(delim)}")
    print(f"Rows loaded: {len(df):,} | Columns: {len(df.columns):,}")
    print(f"Schema overview written to: {out_path}")

if __name__ == "__main__":
    csv_file_path = RAW_DATA / "DevOneIdent_170.csv"
    extract_schema_overview(csv_file_path)