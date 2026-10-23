from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "day.csv"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_CATEGORICAL_RANGES = {
    "season": set(range(1, 5)),
    "mnth": set(range(1, 13)),
    "weekday": set(range(0, 7)),
    "weathersit": set(range(1, 5)),
    "yr": {0, 1},
    "holiday": {0, 1},
    "workingday": {0, 1},
}

def main():
    if not RAW.exists():
        raise FileNotFoundError("Raw data not found. Run `python src/download_data.py` first.")
    df = pd.read_csv(RAW)
    profile = {
        "source_file": str(RAW.relative_to(ROOT)),
        "rows_before": int(len(df)),
        "columns_before": int(len(df.columns)),
        "columns": list(df.columns),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }
    if "dteday" not in df.columns or "cnt" not in df.columns:
        raise ValueError("Unexpected schema: required columns dteday and cnt are missing.")
    df["dteday"] = pd.to_datetime(df["dteday"], errors="raise")
    df = df.sort_values("dteday").reset_index(drop=True)

    # Validate before modifying rows. Do not silently remove outliers.
    if (df["cnt"] < 0).any():
        raise ValueError("Negative target counts found; investigate source data.")
    for col, allowed in EXPECTED_CATEGORICAL_RANGES.items():
        if col in df.columns:
            observed = set(df[col].dropna().unique())
            invalid = observed - allowed
            if invalid:
                raise ValueError(f"Invalid values in {col}: {sorted(invalid)}")
    duplicate_dates = int(df["dteday"].duplicated().sum())
    profile["duplicate_dates"] = duplicate_dates
    profile["date_min"] = str(df["dteday"].min().date())
    profile["date_max"] = str(df["dteday"].max().date())
    profile["rows_after_sort"] = int(len(df))
    profile["missing_after_load"] = {k: int(v) for k, v in df.isna().sum().items()}

    # Keep rows intact; no imputation is applied unless a project-specific decision is documented.
    # Exact duplicates are flagged for review rather than silently dropped.
    df.to_csv(OUT / "day_cleaned.csv", index=False)
    (OUT / "data_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(json.dumps(profile, indent=2))
    print(f"Cleaned table written to {OUT / 'day_cleaned.csv'}")

if __name__ == "__main__":
    main()
