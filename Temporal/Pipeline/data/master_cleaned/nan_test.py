import pandas as pd

df = pd.read_csv("/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/master_cleaned/final_master_cleaned.csv")

nan_counts = df.isna().sum().sort_values(ascending=False)
coverage = (1 - df.isna().sum() / len(df)).sort_values(ascending=False)

with pd.option_context('display.max_rows', None):
    print("Missing Counts:")
    print(nan_counts)
    print("\nCoverage:")
    print(coverage)
