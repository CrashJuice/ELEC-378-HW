import numpy as np

#  Load the data from data.npy
data = np.load('data.npy', allow_pickle=True).item()
X = data['X']  
y = data['y']  

# Add a column of ones at the end of X
X = np.hstack((X, np.ones((X.shape[0], 1))))

# : Estimate using least squares
w_star = np.linalg.inv(X.T @ X) @ X.T @ y

#  Round the weights to the nearest integers
w_rounded = np.round(w_star).astype(int)

#  Decode the message using the A1Z26 cipher using ASCII
message = "".join(" " if num==0 else chr(num+64) for num in w_rounded)

print("Decoded Message:", message)