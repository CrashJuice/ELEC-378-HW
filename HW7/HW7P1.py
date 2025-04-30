import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
# Load dataset
digits = load_digits()
X = digits.data  
y = digits.target  
#  20% of data into test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def compute_distance(x1, x2, p):
    if p == 1:
        return np.sum(np.abs(x1 - x2))
    elif p == 2:
        return np.sqrt(np.sum((x1 - x2) ** 2))
    elif p == np.inf:
        return np.max(np.abs(x1 - x2))
def knn_predict(X_train, y_train, x_test, k, p):
    distances = [compute_distance(x_test, x_train, p) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    return Counter(k_labels).most_common(1)[0][0]
def evaluate_knn(X_train, y_train, X_test, y_test, k_list, p):
    error_rates = []
    for k in k_list:
        errors = 0
        for i, x in enumerate(X_test):
            y_pred = knn_predict(X_train, y_train, x, k, p)
            if y_pred != y_test[i]:
                errors += 1
        error_rate = errors / len(y_test)
        error_rates.append(error_rate)
        print(f"[p={p}] K={k}, Misclassification Rate: {error_rate:.4f}")
    return error_rates
k_values = [1, 3, 5, 7, 9]
# 2 Norm
print("2 Norm")
error_l2 = evaluate_knn(X_train, y_train, X_test, y_test, k_values, p=2)
# 1 Norm
print("1 Norm")
error_l1 = evaluate_knn(X_train, y_train, X_test, y_test, k_values, p=1)
# ∞ Norm 
print("Infinity Norm")
error_inf = evaluate_knn(X_train, y_train, X_test, y_test, k_values, p=np.inf)
plt.plot(k_values, error_l2, marker='o', label='L2 Norm')
plt.plot(k_values, error_l1, marker='s', label='L1 Norm')
plt.plot(k_values, error_inf, marker='^', label='Infinity Norm')
plt.xlabel("K")
plt.ylabel("Misclassification Rate")
plt.title("KNN Misclassification Rate vs. K")
plt.legend()
plt.grid(True)
plt.show()
