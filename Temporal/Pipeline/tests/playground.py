import pandas as pd

df = pd.read_csv("/Users/jbalkovec/Desktop/MDR/Pipeline/data/processed/final.csv")

print("NDVI coverage:", df["NDVI"].notna().mean())
print("LST coverage:", df["LST"].notna().mean())
print("Rain_sat coverage:", df["Rain_sat"].notna().mean())

print(df.loc[df["Rain_sat"].notna(), ["date","Rain_sat"]].head())
