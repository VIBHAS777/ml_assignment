import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

air_quality_data = pd.read_excel("AirQualityUCI.xlsx")
column_name = 'CO(GT)'
clean_data = air_quality_data[column_name].dropna()

mean_co = clean_data.mean()
var_co = clean_data.var(ddof=0)

print(f"Mean of {column_name}: {mean_co}")
print(f"Variance of {column_name}: {var_co}")

plt.figure(figsize=(8, 5))
plt.hist(clean_data, bins=20, color='lightgreen', edgecolor='darkgreen')
plt.title(f'Distribution of {column_name}')
plt.xlabel(column_name)
plt.ylabel('Count')
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

