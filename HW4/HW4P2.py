import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load the data
eaves_file = loadmat('eavesdropping.mat')
eaves_mat = eaves_file["Y"]

# Step 1: Center the data by subtracting the mean of each column
centered_eaves = eaves_mat - np.mean(eaves_mat, axis=0)

# Step 2: Compute SVD
eaves_u, eaves_s, eaves_vh = np.linalg.svd(centered_eaves, full_matrices=False)

# Step 3: Take the first two principal components
eaves_v_approx = eaves_vh[:2, :]  # First two rows of V^H (top 2 principal components)

eaves_pca = centered_eaves @ eaves_v_approx.T  


plt.scatter(eaves_pca[:, 0], eaves_pca[:, 1])
plt.title("PCA Unlabeled Scatter Plot")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()



# create a vector of binary labels
#indicating which cluster each data point lives in
assignment = np.where(eaves_pca[:,0] > np.median(eaves_pca[:,0]), 1, 0)

# Color the Data - feel free to choose whatever colors you want
colors = np.choose(assignment, ['#1f77b4', '#ff7f0e'])

plt.scatter(eaves_pca[:,0], eaves_pca[:, 1], c=colors)
plt.title("PCA Labelled Scatter Plot")
plt.show()






# first principal component (v1) from SVD
v1 = eaves_vh[0, :]

# 1D projection of the centered data
eaves_1d_pca = centered_eaves @ v1.T

# 1D PCA projection
plt.scatter(eaves_1d_pca, np.zeros_like(eaves_1d_pca), c=colors)
plt.title("1D PCA Projection of Data")
plt.show()

# decode the bits
threshold = np.median(eaves_1d_pca)
decoded_assignment = np.where(eaves_1d_pca > threshold, 1, 0)

# Color the data points based on decoded bits
decoded_colors = np.choose(decoded_assignment, ['#1f77b4', '#ff7f0e'])

plt.scatter(eaves_1d_pca, np.zeros_like(eaves_1d_pca), c=decoded_colors)
plt.title("Decoded 1D PCA Projection with Thresholding")
plt.show()
