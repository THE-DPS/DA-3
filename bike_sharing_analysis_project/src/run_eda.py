from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "day_cleaned.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

def main():
    if not DATA.exists():
        raise FileNotFoundError("Run `python src/prepare_data.py` first.")
    df = pd.read_csv(DATA, parse_dates=["dteday"])
    numeric_summary = df.select_dtypes(include="number").describe().round(3)
    numeric_summary.to_csv(OUT / "descriptive_statistics.csv")
    grouped = df.groupby("season", dropna=False)["cnt"].agg(["count", "mean", "median", "std"]).round(2)
    grouped.to_csv(OUT / "demand_by_season.csv")
    monthly = df.groupby("mnth")["cnt"].agg(["count", "mean", "median"]).round(2)
    monthly.to_csv(OUT / "demand_by_month.csv")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["dteday"], df["cnt"], linewidth=1)
    ax.set(title="Daily Bike Rentals Over Time", xlabel="Date", ylabel="Total daily rentals (cnt)")
    fig.tight_layout(); fig.savefig(FIG / "daily_demand_trend.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["cnt"], bins=30, kde=True, ax=ax)
    ax.set(title="Distribution of Daily Rental Counts", xlabel="Daily rentals (cnt)", ylabel="Number of days")
    fig.tight_layout(); fig.savefig(FIG / "target_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x="season", y="cnt", ax=ax)
    ax.set(title="Daily Rentals by Season Code", xlabel="Season code (see UCI data dictionary)", ylabel="Daily rentals (cnt)")
    fig.tight_layout(); fig.savefig(FIG / "rentals_by_season.png", dpi=160); plt.close(fig)

    for col in ["temp", "atemp", "hum", "windspeed"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=df, x=col, y="cnt", alpha=.65, ax=ax)
        ax.set(title=f"Daily Rentals vs {col}", xlabel=f"{col} (source-normalized)", ylabel="Daily rentals (cnt)")
        fig.tight_layout(); fig.savefig(FIG / f"rentals_vs_{col}.png", dpi=160); plt.close(fig)

    corr_cols = [c for c in ["cnt", "temp", "atemp", "hum", "windspeed", "mnth", "weekday"] if c in df.columns]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[corr_cols].corr(numeric_only=True), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation Matrix (Selected Numeric Fields)")
    fig.tight_layout(); fig.savefig(FIG / "correlation_heatmap.png", dpi=160); plt.close(fig)
    print(f"EDA outputs saved under {OUT}")

if __name__ == "__main__":
    main()
