import pandas as pd
import csv
from pathlib import Path
from dbx.paths import RAW_DATA

def detect_delimiter(csv_path, sample_bytes=20000):
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(sample_bytes)
    dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
    return dialect.delimiter

csv_file_path = RAW_DATA / "DevOneIdent_170.csv"
delim = detect_delimiter(csv_file_path)

df = pd.read_csv(
    csv_file_path,
    sep=delim,
    engine="python",
    encoding="utf-8",
    encoding_errors="replace",
    on_bad_lines="skip"
)

# show language-ish columns
lang_cols = [c for c in df.columns if "lang" in str(c).lower()]
print("Language-related columns:")
for c in lang_cols:
    print(repr(c))

print("\nFirst row values for language-related columns:")
row0 = df.iloc[0].to_dict()
for c in lang_cols:
    print(f"{repr(c)} = {row0.get(c)}")