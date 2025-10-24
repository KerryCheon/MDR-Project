import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

MASTER_CSV = Path("data/master/final_master.csv").resolve()
df = pd.read_csv(MASTER_CSV)

numeric = df.select_dtypes(include="number")
corr = numeric.corr(method="spearman")

plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Feature Correlation Matrix (Spearman)")
plt.tight_layout()
plt.show()
