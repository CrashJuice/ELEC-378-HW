import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================
# PART A: Load and Plot Dataset
# ============================

# Load the first dataset from Train1.mat
data1 = scipy.io.loadmat("Train1.mat")
X = data1['X']  # Feature matrix of shape (100, 2)
y = data1['y'].flatten()  # Labels vector of shape (100,), flattened from column vector

# Randomly initialize weight vector w and bias b
w = np.random.randn(2)  # w has same dimension as features
b = np.random.randn()   # bias term

# Plot data points and a randomly initialized hyperplane
plt.figure(figsize=(6, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class +1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', label='Class -1')

# Compute the decision boundary line based on current w, b
x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
y_vals = -(w[0] * x_vals + b) / w[1]

plt.plot(x_vals, y_vals, 'k--', label='Random Hyperplane')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Part (a): Random Hyperplane on Train1')
plt.legend()
plt.grid(True)
plt.show()


# ===================================================
# Subgradient Computation for Soft-Margin SVM (Part B)
# ===================================================

# Computes the subgradient of the soft-margin SVM loss
def subgradient_step(X, y, w, b, lam):
    n = len(y)
    grad_w = np.zeros_like(w)
    grad_b = 0.0

    for i in range(n):
        margin = y[i] * (np.dot(w, X[i]) + b)
        if margin < 1:
            # Only count samples violating margin
            grad_w -= y[i] * X[i]  
            grad_b -= y[i]

    # Add L2 regularization gradient
    grad_w += 2 * lam * w
    return grad_w, grad_b


# Gradient descent training for soft-margin SVM
def svm_gradient_descent(X, y, lam=1.0, lr=0.01, epochs=100):
    p = X.shape[1]
    w = np.random.randn(p)
    b = np.random.randn()
    history = [(w.copy(), b)]  # Store history for visualization

    for _ in range(epochs):
        grad_w, grad_b = subgradient_step(X, y, w, b, lam)
        w -= lr * grad_w
        b -= lr * grad_b
        history.append((w.copy(), b))
    
    return w, b, history


# ========================
# Plotting SVM Progression
# ========================

# Plot hyperplane evolution during training
def plot_progression(X, y, history, step=10, title="Hyperplane Progression"):
    plt.figure(figsize=(7, 7))
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class +1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', label='Class -1')
    
    x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)

    # Plot intermediate hyperplanes at intervals
    for i in range(0, len(history), step):
        w, b = history[i]
        if w[1] != 0:
            y_vals = -(w[0] * x_vals + b) / w[1]
            alpha = 0.2 + 0.8 * (i / len(history))  # Fade earlier lines
            plt.plot(x_vals, y_vals, linestyle='--', color='gray', alpha=alpha)
    
    # Final trained hyperplane
    w_final, b_final = history[-1]
    y_final = -(w_final[0] * x_vals + b_final) / w_final[1]
    plt.plot(x_vals, y_final, 'k-', label='Final Hyperplane')

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run training and visualize learning
w_final, b_final, history = svm_gradient_descent(X, y, lam=1.0, lr=0.01, epochs=100)
plot_progression(X, y, history, step=5, title="(b) Hyperplane Progression for λ=1")


# =========================================
# PART C: Try Multiple λ Values with GD
# =========================================

lambda_values = [0.01, 0.1, 10.0, 100.0]
titles = [f"(c) λ = {lam}" for lam in lambda_values]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)

# Plot hyperplane evolution for different regularization values
for i, lam in enumerate(lambda_values):
    w_final, b_final, history = svm_gradient_descent(X, y, lam=lam, lr=0.01, epochs=100)

    ax = axes[i]
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class +1')
    ax.scatter(X[y == -1, 0], X[y == -1, 1], color='red', label='Class -1')

    for j in range(0, len(history), 10):
        w, b = history[j]
        if w[1] != 0:
            y_vals = -(w[0] * x_vals + b) / w[1]
            alpha = 0.2 + 0.8 * (j / len(history))
            ax.plot(x_vals, y_vals, linestyle='--', color='gray', alpha=alpha)

    # Final hyperplane
    w_final, b_final = history[-1]
    y_final = -(w_final[0] * x_vals + b_final) / w_final[1]
    ax.plot(x_vals, y_final, 'k-', label='Final Hyperplane')

    ax.set_title(titles[i])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_ylim(-5, 5)
    ax.grid(True)
    ax.legend()

plt.suptitle("Part (c): Effect of λ on SVM Hyperplane Convergence", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# ======================================
# PART D: Stochastic Gradient Descent
# ======================================

# SGD version of the SVM trainer
def svm_sgd(X, y, lam=1.0, lr=0.01, epochs=100):
    n, p = X.shape
    w = np.random.randn(p)
    b = np.random.randn()
    history = [(w.copy(), b)]

    for _ in range(epochs * n):  # One update per data point per epoch
        i = np.random.randint(n)  # Pick a random sample
        xi = X[i]
        yi = y[i]
        margin = yi * (np.dot(w, xi) + b)

        # Compute subgradients depending on margin
        if margin < 1:
            grad_w = -yi * xi + 2 * lam * w
            grad_b = -yi
        else:
            grad_w = 2 * lam * w
            grad_b = 0

        # Update parameters
        w -= lr * grad_w
        b -= lr * grad_b

        # Save state once per epoch
        if _ % n == 0:
            history.append((w.copy(), b))
    return w, b, history

# Compare SGD results for different λ values
lambda_values = [0.01, 0.1, 10.0, 100.0]
titles = [f"(d) λ = {lam} (SGD)" for lam in lambda_values]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)

for i, lam in enumerate(lambda_values):
    w_final, b_final, history = svm_sgd(X, y, lam=lam, lr=0.01, epochs=100)

    ax = axes[i]
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class +1')
    ax.scatter(X[y == -1, 0], X[y == -1, 1], color='red', label='Class -1')

    for j in range(0, len(history), 2):
        w, b = history[j]
        if w[1] != 0:
            y_vals = -(w[0] * x_vals + b) / w[1]
            alpha = 0.2 + 0.8 * (j / len(history))
            ax.plot(x_vals, y_vals, linestyle='--', color='gray', alpha=alpha)

    # Final hyperplane
    w_final, b_final = history[-1]
    y_final = -(w_final[0] * x_vals + b_final) / w_final[1]
    ax.plot(x_vals, y_final, 'k-', label='Final Hyperplane')

    ax.set_title(titles[i])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_ylim(-5, 5)
    ax.grid(True)
    ax.legend()

plt.suptitle("Part (d): SGD Convergence for Different λ", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# ============================================
# PART E: Evaluate on a New Dataset (Train2.mat)
# ============================================

# Load second dataset
data2 = scipy.io.loadmat("Train2.mat")
X2 = data2['X']
y2 = data2['y'].flatten()

# Train with original gradient descent and λ=1
w2, b2, history2 = svm_gradient_descent(X2, y2, lam=1.0, lr=0.01, epochs=100)

# Simple prediction function using final weights
def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)

# Compute training error
y2_pred = predict(X2, w2, b2)
error2 = np.mean(y2_pred != y2)  # classification error rate

print("Part (e): Classification Error on Train2 =", error2)
