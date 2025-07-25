import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

mean = 0
std_dev = 1
sample_data = np.random.normal(loc=mean, scale=std_dev, size=1000)

fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(sample_data, bins=30, stat='density', color='lightcoral', edgecolor='black', label='Sample Histogram', ax=ax)

x_values = np.linspace(sample_data.min(), sample_data.max(), 1000)
normal_pdf = norm.pdf(x_values, loc=mean, scale=std_dev)
ax.plot(x_values, normal_pdf, color='darkblue', linewidth=2, label='Standard Normal PDF')

ax.set_title("Comparison of Histogram and Standard Normal Distribution")
ax.set_xlabel("Random Variable")
ax.set_ylabel("Density")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()

