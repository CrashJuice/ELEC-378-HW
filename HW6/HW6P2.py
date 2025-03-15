import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso

# Load data
mat = scipy.io.loadmat('CS.mat')
y = mat['y']  # Compressive measurements
Phi = mat['Phi']  # Sensing matrix
Psi = mat['Psi']  # Sparsifying matrix

N = y.shape[0]  # Number of measurements

# (a) Solve for s directly since Phi * Psi is square
s = np.linalg.inv(Phi @ Psi) @ y
x = Psi @ s  # Reconstruct the image
x = x.reshape((64, 64))

# Display the ground truth image
plt.imshow(x, cmap='gray')
plt.title("Ground Truth Image")
plt.show()

# (b) Compressive Sampling (Randomly selecting N/2 rows)
M = N // 2
indices = np.random.choice(N, M, replace=False)
Phic = Phi[indices, :]
yc = y[indices]

# Ridge Regression to recover s
ridge = Ridge(alpha=0.1)
ridge.fit(Phic @ Psi, yc)
s_ridge = ridge.coef_.reshape(-1, 1)
x_ridge = Psi @ s_ridge
x_ridge = x_ridge.reshape((64, 64))

# Display Ridge Recovery
plt.imshow(x_ridge, cmap='gray')
plt.title("Ridge Regression Recovery")
plt.show()

# (c) Lasso Regression for sparse recovery
lasso = Lasso(alpha=0.01)
lasso.fit(Phic @ Psi, yc.ravel())
s_lasso = lasso.coef_.reshape(-1, 1)
x_lasso = Psi @ s_lasso
x_lasso = x_lasso.reshape((64, 64))

# Display Lasso Recovery
plt.imshow(x_lasso, cmap='gray')
plt.title("Lasso Regression Recovery")
plt.show()

# (d) Constructing a K-sparse version of s
K = 100  # Set a threshold for sparsity
sK = np.where(np.abs(s) > 15, s, 0)
xK = Psi @ sK
xK = xK.reshape((64, 64))

# Display Sparse Image
plt.imshow(xK, cmap='gray')
plt.title("Sparse Image xK")
plt.show()
