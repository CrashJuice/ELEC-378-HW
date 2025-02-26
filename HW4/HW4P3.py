import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load data from MATLAB file
cancer_data = loadmat('cancer.mat')
X = cancer_data["X"]  # shape: (64 patients, 6830 genes)
Y = cancer_data["Y"]  # shape: (64 patients, ...)

# ===============================
# Part 1: PCA using SVD (2D projection) and Visualization by Diagnosis
# ===============================

# Step 1: Center the data 
X_c = X - np.mean(X, axis=0)

# Step 2: Compute SVD
# X = U * S * Vh, where rows of Vh are the principal directions
X_u, X_s, X_vh = np.linalg.svd(X_c, full_matrices=False)

# Step 3: Take the first two principal components (largest singular values)
X_v_approx = X_vh[:2, :]  

# Step 4: Project the centered data onto the 2D space
X_pca = X_c @ X_v_approx.T   # resulting shape: (64, 2)

# Process the labels: assume each label is stored in a nested array; clean them up
cleaned_labels = [tag[0][0].strip() for tag in Y]

# Create an assignment: 1 for MELANOMA, 0 for Others
assignment = np.where(np.array([label == 'MELANOMA' for label in cleaned_labels]), 1, 0)

# Color mapping: Blue for Other, Orange for Melanoma
colors = np.choose(assignment, ['#1f77b4', '#ff7f0e'])

# Scatter plot of the 2D PCA projection by diagnosis
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors)
plt.title("PCA Projection: Melanoma vs. Other")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
melanoma_patch = mpatches.Patch(color='#ff7f0e', label="Melanoma")
other_patch = mpatches.Patch(color='#1f77b4', label="Other")
plt.legend(handles=[other_patch, melanoma_patch])
plt.show()

# ===============================
# Part 2: Heatmap of Top Contributing Genes
# ===============================

# Use the second principal component (or change index as desired) for gene contributions.
# (Typically, the first PC is most informative; adjust index if needed)
vj = X_vh[1, :]  # Principal direction for gene contributions
sorter = np.argsort(np.abs(vj))[::-1]  # Sort gene indices by absolute contribution (largest first)

num_columns = 50  # Number of top contributing genes to display
Xsort = X[:, sorter[:num_columns]]  # Subset of X with top contributing genes

plt.figure(figsize=(10,6))
plt.imshow(Xsort, aspect='auto', cmap='hot')
plt.title("Heatmap of Top Contributing Genes for Melanoma Diagnosis")
plt.xlabel("Top Contributing Genes")
plt.ylabel("Patients")
plt.colorbar(label="Expression Level")
plt.show()

# ===============================
# Part 3: KMeans Clustering in Higher Dimensions (Using sklearn)
# ===============================

# Perform PCA retaining 3 components (using sklearn this time)
pca = PCA(n_components=3)
X_pca_3d = pca.fit_transform(X)  # shape: (64, 3)

# Run KMeans on the 3D PCA-transformed data, using 3 clusters
kmeans = KMeans(n_clusters=3, n_init=10, max_iter=300, random_state=42)
kmeans.fit(X_pca_3d)
labels = kmeans.labels_

# For visualization, project the 3D data to 2D by taking the first two principal components
X_pca_2d = X_pca_3d[:, :2]

# Plot the 2D projection with points colored by their KMeans cluster assignment
plt.figure(figsize=(8,6))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=labels, cmap='viridis', s=50)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("KMeans Clustering (3 clusters) on PCA (3 PCs)")
plt.show()
