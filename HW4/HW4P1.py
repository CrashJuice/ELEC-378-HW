import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.image as im

# Load and display the original image
image = im.imread('objection.png')
plt.imshow(image)
plt.title('Original Image')
plt.show()

# Form the data matrix from the RGB image
X = image.reshape(-1, 3)  # Flatten the image into an (n, 3) matrix, where n = total pixels

# Run PCA
pca = PCA(n_components=2)  # Reduce to 2D

# Project pixels into 2D space
pixels_transformed = pca.fit_transform(X)

# Plot pixels in 2D space, colored by their original RGB values
plt.scatter(pixels_transformed[:, 0], pixels_transformed[:, 1], c=X / 255, s=1)  # Normalize RGB values to [0,1]
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('PCA Projection of Image Colors')
plt.show()


import numpy as np

def kmeans(X, k=64, max_iterations=100):
   
  
    X = np.array(X)

    random_indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[random_indices]

    for _ in range(max_iterations):
        # Compute Euclidean distances from each point to the centroids
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Shape: (num_points, k)

        # Assign each point to the closest centroid
        P = np.argmin(distances, axis=1)

        # Compute new centroids
        new_centroids = np.array([X[P == i].mean(axis=0) if np.any(P == i) else centroids[i] for i in range(k)])

        # Stop if centroids don't change
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return P, centroids

#Using K means we created:
K = 64

labels, centroids = kmeans(X, k=K, max_iterations=100)


color_quantized_data_matrix = np.vstack(centroids[labels])  


plt.imshow(color_quantized_data_matrix.reshape(image.shape))
plt.title('Quantized Image with K-means')
plt.show()




#Using Sklearn:

kmeans = KMeans(n_clusters=K, n_init=10, max_iter=100, random_state=42).fit(X)


labels = kmeans.labels_
kmeans_flat = kmeans.cluster_centers_[labels]  


plt.imshow(kmeans_flat.reshape(image.shape))
plt.title('Quantized Image (SKLearn K-means)')
plt.show()


#Using our labels to plot vernollis tiling onto the PCA graph

plt.scatter(pixels_transformed[:, 0], pixels_transformed[:, 1], c=color_quantized_data_matrix, s=1)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('PCA Projection with Quantized Colors')
plt.show()