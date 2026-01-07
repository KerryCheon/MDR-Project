import pandas as pd

df = pd.read_csv(
    "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/master_cleaned/final_master_cleaned.csv"
)

n_rows = len(df)

nan_counts = df.isna().sum()
coverage = 1 - nan_counts / n_rows

summary = (
    pd.DataFrame({
        "missing_count": nan_counts,
        "coverage_pct": (coverage * 100).round(2)
    })
    .sort_values("missing_count", ascending=False)
)

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"
BOLD = "\033[1m"

def color_coverage(val):
    if val < 50:
        return RED
    elif val < 90:
        return YELLOW
    else:
        return GREEN

print(f"\n{BOLD}Dataset overview{RESET}")
print(f"Rows: {n_rows:,}")
print(f"Columns: {df.shape[1]}\n")

print(f"{BOLD}Missingness & coverage by column{RESET}")

for col, row in summary.iterrows():
    color = color_coverage(row["coverage_pct"])
    print(
        f"{col:<35} "
        f"missing: {row['missing_count']:>6} | "
        f"{color}coverage: {row['coverage_pct']:>6.2f}%{RESET}"
    )
