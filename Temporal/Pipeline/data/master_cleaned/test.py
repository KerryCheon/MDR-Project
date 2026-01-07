import pandas as pd

df = pd.read_csv("final_master_cleaned.csv", low_memory=False)
n = len(df)

# pick a few "signature" columns for each schema
sig_sat = ["s2_b2", "s1_vv", "NDVI_modis", "LST_modis"]
sig_snotel = ["DateTime", "SM_prev", "Rain_3d", "elevation"]  # from your printout

def any_present(cols):
    cols = [c for c in cols if c in df.columns]
    return df[cols].notna().any(axis=1) if cols else pd.Series([False]*n)

has_sat = any_present(sig_sat)
has_snotel = any_present(sig_snotel)

print("Rows total:", n)
print("Rows with satellite-signature:", int(has_sat.sum()))
print("Rows with snotel-signature:", int(has_snotel.sum()))
print("Rows with both:", int((has_sat & has_snotel).sum()))
print("Rows with neither:", int((~has_sat & ~has_snotel).sum()))
