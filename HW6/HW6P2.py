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

#A Solve for s directly since Phi * Psi is square
s = np.linalg.inv(Phi @ Psi) @ y
x = Psi @ s  
x = x.reshape((64, 64))

#plt.imshow(x, cmap='gray')
#plt.title("Ground Truth Image")
#plt.show()

#B Compressive Sampling (Randomly selecting N/2 rows)
M = N // 2
indices = np.random.choice(N, M, replace=False)
Phic = Phi[indices, :]
yc = y[indices]

# Ridge Regression to get s
ridge = Ridge(alpha=10)
ridge.fit(Phic @ Psi, yc)
s_ridge = ridge.coef_.reshape(-1, 1)
x_ridge = Psi @ s_ridge
x_ridge = x_ridge.reshape((64, 64))


#plt.imshow(x_ridge, cmap='gray')
#plt.title("Ridge Regression Recovery")
#plt.show()

#(c) Lasso Regression for sparse recovery
lasso = Lasso(alpha=0.01)
lasso.fit(Phic @ Psi, yc.ravel())
s_lasso = lasso.coef_.reshape(-1, 1)
x_lasso = Psi @ s_lasso
x_lasso = x_lasso.reshape((64, 64))

# Display Lasso Recovery
plt.imshow(x_lasso, cmap='gray')
plt.title("Lasso Regression Recovery")
plt.show()



# Create sparse s_K where only values above a threshold are kept
threshold = 15
sK = np.where(np.abs(s) > threshold, s, 0)

#  sparse image xK
xK = Psi @ sK
xK = xK.reshape((64, 64))


plt.imshow(xK, cmap='gray')
plt.title("Sparse Image xK")
plt.show()

# Define K as the number of nonzero elements in sK
K = np.count_nonzero(sK)
M = 2 * K  # Theoretical minimum 

# Randomly select M rows for compressive sampling
indices = np.random.choice(N, M, replace=False)
Phic = Phi[indices, :]
yc = y[indices]

# Ridge Regression 
ridge = Ridge(alpha=0.1)
ridge.fit(Phic @ Psi, yc)
s_ridgeK = ridge.coef_.reshape(-1, 1)
x_ridgeK = Psi @ s_ridgeK
x_ridgeK = x_ridgeK.reshape((64, 64))

# Display Ridge 
plt.imshow(x_ridgeK, cmap='gray')
plt.title(f"Ridge Recovery with M=2K")
plt.show()

# Lasso Regression
lasso = Lasso(alpha=0.01)
lasso.fit(Phic @ Psi, yc.ravel())
s_lassoK = lasso.coef_.reshape(-1, 1)
x_lassoK = Psi @ s_lassoK
x_lassoK = x_lassoK.reshape((64, 64))

# Display Lasso 
plt.imshow(x_lassoK, cmap='gray')
plt.title(f"Lasso Recovery with M=2K")
plt.show()
