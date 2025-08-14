import numpy as np
import pandas as pd
import math
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib.colors import ListedColormap

# Load dataset
df = pd.read_excel("AirQualityUCI.xlsx")

# Remove rows with missing placeholder values (-200)
df = df[df['CO(GT)'] > -200]
df = df[df['C6H6(GT)'] > -200]
df = df[df['T'] > -200]

# Function for equal-width binning
def equal_width_binning(series, bins=4):
    min_val, max_val = series.min(), series.max()
    bins_edges = np.linspace(min_val, max_val, bins + 1)
    binned = np.digitize(series, bins_edges, right=False) - 1
    binned[binned == bins] = bins - 1   # adjust values at upper edge
    return binned

# Function to calculate entropy
def entropy(y):
    counts = np.bincount(y)
    probs = counts / len(y)
    return -np.sum([p * math.log2(p) for p in probs if p > 0])

# Function to calculate gini index
def gini_index(y):
    counts = np.bincount(y)
    probs = counts / len(y)
    return 1 - np.sum([p ** 2 for p in probs])

# Function to calculate information gain
def information_gain(X_col, y):
    base_entropy = entropy(y)
    values, counts = np.unique(X_col, return_counts=True)
    weighted_entropy = 0
    for v, count in zip(values, counts):
        weighted_entropy += (count / len(X_col)) * entropy(y[X_col == v])
    return base_entropy - weighted_entropy

# Function to select best feature for root node
def select_root_node(X, y):
    best_gain = -1
    best_feat = None
    for col in X.columns:
        # bin if feature is continuous
        if not pd.api.types.is_integer_dtype(X[col]):
            feature_vals = equal_width_binning(X[col])
        else:
            feature_vals = X[col]
        gain = information_gain(feature_vals, y)
        print(f"Info Gain for feature {col}: {gain}")
        if gain > best_gain:
            best_gain = gain
            best_feat = col
    return best_feat, best_gain

# Simple decision tree class (ID3-like)
class SimpleDecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        # stop if pure class, no features left, or max depth reached
        if entropy(y) == 0 or len(X.columns) == 0 or depth == self.max_depth:
            return Counter(y).most_common(1)[0][0]
        best_feat, _ = select_root_node(X, y)
        if best_feat is None:
            return Counter(y).most_common(1)[0][0]
        node = {best_feat: {}}
        feat_values = equal_width_binning(X[best_feat])
        for v in np.unique(feat_values):
            idx = feat_values == v
            if idx.sum() == 0:
                node[best_feat][v] = Counter(y).most_common(1)[0][0]
            else:
                X_sub = X.loc[idx].drop(columns=[best_feat])
                y_sub = y[idx]
                node[best_feat][v] = self.fit(X_sub, y_sub, depth + 1)
        self.tree = node
        return node

    def predict_one(self, x, tree=None):
        tree = tree or self.tree
        if not isinstance(tree, dict):
            return tree
        feat = next(iter(tree))
        val = x[feat]
        bin_val = np.digitize([val], np.linspace(x[feat].min(), x[feat].max(), 5))[0] - 1
        return self.predict_one(x, tree[feat].get(bin_val, Counter(tree[feat]).most_common(1)[0][0]))

# Prepare data (two features, binary classification on C6H6(GT))
n_bins = 2
X = df[['CO(GT)', 'T']]
y = equal_width_binning(df['C6H6(GT)'], bins=n_bins)

X_binned = pd.DataFrame({
    'CO(GT)': equal_width_binning(X['CO(GT)']),
    'T': equal_width_binning(X['T'])
})

# Print entropy and gini index
print("1. Entropy of the output:", entropy(y))
print("2. Gini index of the output:", gini_index(y))

# Select best root feature
selected_feature, gain = select_root_node(X_binned, y)
print(f"3. Feature selected as root node: {selected_feature} (Information Gain: {gain})")

# Build tree
tree = SimpleDecisionTree(max_depth=3)
model = tree.fit(X_binned, y)
print("5. Decision Tree (nested dictionary):")
print(model)

# Visualize sklearn decision tree
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_binned, y)
plt.figure(figsize=(10,6))
plot_tree(clf, feature_names=['CO(GT)', 'T'], class_names=['Low', 'High'], filled=True)
plt.title("6. Decision Tree Visualization")
plt.show()

# Visualize decision boundary
X_plot, y_plot = X_binned.values, y
x_min, x_max = X_plot[:, 0].min()-1, X_plot[:, 0].max()+1
y_min, y_max = X_plot[:, 1].min()-1, X_plot[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), alpha=0.6)
plt.scatter(X_plot[:,0], X_plot[:,1], c=y_plot, cmap=ListedColormap(['#FF0000','#0000FF']), edgecolor='k')
plt.xlabel('CO(GT) (binned)')
plt.ylabel('Temperature (binned)')
plt.title('7. Decision Boundary for Decision Tree')
plt.show()

