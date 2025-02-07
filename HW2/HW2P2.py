import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image = Image.open("valeriacute.jpg").convert("L")  
A = np.array(image, dtype=float)


U, S, Vt = np.linalg.svd(A, full_matrices=False)

#singular values plotted
plt.figure(figsize=(8, 5))
plt.plot(S, marker="o")
plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.title("Singular Value order")
plt.show()

# (b) Decide on low-rank approximation
# I think the image can be well approximated by a low rank matrix because the values drop off really quickly 
#The biggest singular value is ~80,000 and the 4th biggest is only ~10,000 and continues to rapidly decrease

# (c) Low-rank approximation with rank-k
#regular image first

plt.figure(figsize=(8, 8))
plt.imshow(A, cmap="gray")
plt.title(f"real grayscale")
plt.axis("off")
plt.show()

k = 10  # rank value
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

plt.figure(figsize=(8, 8))
plt.imshow(A_approx, cmap="gray")
plt.title(f"Low-Rank Approximation (k={k})")
plt.axis("off")
plt.show()