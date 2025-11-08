import pandas as pd
import plotly.express as px

CSV_PATH = "/Users/jbalkovec/Desktop/MDR/Temporal/Pipeline/data/master/final_master.csv"
df = pd.read_csv(CSV_PATH)
df.drop(columns=["station_id", "crx_vn", "longitude", "latitude", "SM_label",
                 "soil_moisture_10cm", "soil_moisture_20cm", "soil_moisture_50cm", "soil_moisture_100cm",
                 "soil_temp_10cm", "soil_temp_20cm", "soil_temp_50cm", "soil_temp_100cm"], inplace=True, errors='ignore')

corr = df.corr(numeric_only=True)

fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    title="Interactive Correlation Heatmap"
)
fig.show()
