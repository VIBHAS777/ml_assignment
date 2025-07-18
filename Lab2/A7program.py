import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_all_similarity(vec1_bin, vec2_bin, vec1_full, vec2_full):
    f11 = np.sum(vec1_bin & vec2_bin)
    f00 = np.sum(~vec1_bin & ~vec2_bin)
    f01 = np.sum(~vec1_bin & vec2_bin)
    f10 = np.sum(vec1_bin & ~vec2_bin)
    jc = f11 / (f01 + f10 + f11)
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)
    cos = np.dot(vec1_full, vec2_full) / (np.linalg.norm(vec1_full) * np.linalg.norm(vec2_full))
    return (jc + smc + cos) / 3

file_path = "Lab Session Data.xlsx"
thyroid_df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

df_numeric = thyroid_df.select_dtypes(include=[np.number])
df_bin = df_numeric.astype(bool)

sample_size = 20
sim_matrix = np.zeros((sample_size, sample_size))

for i in range(sample_size):
    for j in range(sample_size):
        vec1_bin = df_bin.iloc[i]
        vec2_bin = df_bin.iloc[j]
        vec1_full = df_numeric.iloc[i].values
        vec2_full = df_numeric.iloc[j].values
        sim_matrix[i, j] = calculate_all_similarity(vec1_bin, vec2_bin, vec1_full, vec2_full)

plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("JC + SMC + Cosine Similarity (Averaged) Heatmap for First 20 Vectors")
plt.xlabel("Sample Index")
plt.ylabel("Sample Index")
plt.tight_layout()
plt.show()
