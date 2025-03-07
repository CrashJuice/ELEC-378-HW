from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(int)  # Convert labels to integers

# Run K-Means clustering with 10 clusters
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

misclassified_counts = {}

# For each digit, determine its majority cluster and count misclassifications
for digit in range(10):
    # Get the indices of images that correspond to the current digit
    indices = np.where(y == digit)[0]
    # Retrieve the cluster labels for these images
    clusters_for_digit = cluster_labels[indices]
    # Identify the majority cluster for the current digit
    majority_cluster = Counter(clusters_for_digit).most_common(1)[0][0]
    # Count how many images of this digit are not in the majority cluster
    misclassified_counts[digit] = np.sum(clusters_for_digit != majority_cluster)
    # Print the majority cluster for the digit
    print(f"Digit {digit}: Majority cluster = {majority_cluster}")

print("\nMisclassified counts for each digit:")
for digit, count in misclassified_counts.items():
    print(f"Digit {digit}: {count} misclassified")