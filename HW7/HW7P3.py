import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Load data
data1 = scipy.io.loadmat("Train1.mat")
X = data1['X']  # shape (100, 2)
y = data1['y'].flatten()  # shape (100,)


# Randomly initialize weight vector w and bias b
w = np.random.randn(2)
b = np.random.randn()

# Plot the data and the random hyperplane
plt.figure(figsize=(6, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class +1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', label='Class -1')

# Decision boundary: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b)/w2
x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
y_vals = -(w[0] * x_vals + b) / w[1]

plt.plot(x_vals, y_vals, 'k--', label='Random Hyperplane')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Part (a): Random Hyperplane on Train1')
plt.legend()
plt.grid(True)
plt.show()

# Gradient of soft-margin SVM objective
def subgradient_step(X, y, w, b, lam):
    n = len(y)
    grad_w = np.zeros_like(w)
    grad_b = 0.0

    for i in range(n):
        margin = y[i] * (np.dot(w, X[i]) + b)
        if margin < 1:
            grad_w -= y[i] * X[i]  # hinge loss active
            grad_b -= y[i]
    
    grad_w += 2 * lam * w  # add regularization term
    return grad_w, grad_b

# Gradient Descent Loop
def svm_gradient_descent(X, y, lam=1.0, lr=0.01, epochs=100):
    p = X.shape[1]
    w = np.random.randn(p)
    b = np.random.randn()
    history = [(w.copy(), b)]  # track (w, b) at each step

    for _ in range(epochs):
        grad_w, grad_b = subgradient_step(X, y, w, b, lam)
        w -= lr * grad_w
        b -= lr * grad_b
        history.append((w.copy(), b))
    
    return w, b, history

# Plotting function for progression
def plot_progression(X, y, history, step=10, title="Hyperplane Progression"):
    plt.figure(figsize=(7, 7))
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class +1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', label='Class -1')
    
    x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)

    for i in range(0, len(history), step):
        w, b = history[i]
        if w[1] != 0:
            y_vals = -(w[0] * x_vals + b) / w[1]
            alpha = 0.2 + 0.8 * (i / len(history))
            plt.plot(x_vals, y_vals, linestyle='--', color='gray', alpha=alpha)
    
    # Final hyperplane in black
    w_final, b_final = history[-1]
    y_final = -(w_final[0] * x_vals + b_final) / w_final[1]
    plt.plot(x_vals, y_final, 'k-', label='Final Hyperplane')

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run SVM gradient descent and plot progression
w_final, b_final, history = svm_gradient_descent(X, y, lam=1.0, lr=0.01, epochs=100)
plot_progression(X, y, history, step=5, title="(b) Hyperplane Progression for λ=1")

# Part (c): Compare multiple lambda values in a 2x2 grid
lambda_values = [0.01, 0.1, 10.0, 100.0]
titles = [f"(c) λ = {lam}" for lam in lambda_values]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)

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

    # Final line in black
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

#Part D stochastic gradient descent funtion
def svm_sgd(X, y, lam=1.0, lr=0.01, epochs=100):
    n, p = X.shape
    w = np.random.randn(p)
    b = np.random.randn()
    history = [(w.copy(), b)]

    for _ in range(epochs * n):  # do n updates per epoch
        i = np.random.randint(n)
        xi = X[i]
        yi = y[i]
        margin = yi * (np.dot(w, xi) + b)

        # Compute subgradients
        if margin < 1:
            grad_w = -yi * xi + 2 * lam * w
            grad_b = -yi
        else:
            grad_w = 2 * lam * w
            grad_b = 0

        # SGD update
        w -= lr * grad_w
        b -= lr * grad_b

        # Save only every epoch for plotting
        if _ % n == 0:
            history.append((w.copy(), b))
    return w, b, history

# Part (d): SGD version of the 2x2 lambda grid
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

#PART E

# Load Train2.mat
data2 = scipy.io.loadmat("Train2.mat")
X2 = data2['X']
y2 = data2['y'].flatten()

# Train using gradient descent (λ = 1)
w2, b2, history2 = svm_gradient_descent(X2, y2, lam=1.0, lr=0.01, epochs=100)

# Predict labels
def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)

y2_pred = predict(X2, w2, b2)
error2 = np.mean(y2_pred != y2)

# Plot the data and learned hyperplane
plt.figure(figsize=(6, 6))
plt.scatter(X2[y2 == 1, 0], X2[y2 == 1, 1], color='blue', label='Class +1')
plt.scatter(X2[y2 == -1, 0], X2[y2 == -1, 1], color='red', label='Class -1')

x_vals = np.linspace(np.min(X2[:, 0]), np.max(X2[:, 0]), 100)
y_vals = -(w2[0] * x_vals + b2) / w2[1]
plt.plot(x_vals, y_vals, 'k--', label='Learned Hyperplane')

plt.title(f"(e) Train2 with λ = 1\nMisclassification Error: {error2:.2f}")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.legend()
plt.show()

#PART F

data2 = scipy.io.loadmat("Train2.mat")
X2 = data2['X']
y2 = data2['y'].flatten()

# Define nonlinear feature mappings 
def phi_radial(X):
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack((x1, x2, np.sqrt(x1**2 + x2**2)))

def phi_parabola(X):
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack((x1, x2, x1**2 + x2**2))

def phi_saddle(X):
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack((x1, x2, x1**2 - x2**2))

# NEW: Downward-opening parabola feature map
def phi_parabola_open_down(X):
    x1, x2 = X[:, 0], X[:, 1]
    x3 = (x2 + x1**2 - 1)**2
    return np.column_stack((x1, x2, x3))

# List of φ functions to test
phi_funcs = [
    ("(√(x1²+x2²))", phi_radial),
    ("(x1²+x2²)", phi_parabola),
    ("(x1²−x2²)", phi_saddle),
    ("((x2 + x1² − 1)²)", phi_parabola_open_down)
]


# 3D Plotting
for title, phi_func in phi_funcs:
    X_mapped = phi_func(X2)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_mapped[y2 == 1, 0], X_mapped[y2 == 1, 1], X_mapped[y2 == 1, 2], color='blue', label='Class +1')
    ax.scatter(X_mapped[y2 == -1, 0], X_mapped[y2 == -1, 1], X_mapped[y2 == -1, 2], color='red', label='Class -1')

    ax.set_title(f"(f) 3D phi Mapping: {title}")
    ax.set_xlabel("1(x)")
    ax.set_ylabel("2(x)")
    ax.set_zlabel("3(x)")
    ax.legend()
    plt.tight_layout()
    plt.show()

#PART G :P

# Transform data using phi
X2_mapped = phi_radial(X2)

# Train SVM on transformed data
w_phi, b_phi, history_phi = svm_gradient_descent(X2_mapped, y2, lam=1, lr=0.01, epochs=5000)

print("w_phi:", w_phi)
print("b_phi:", b_phi)


# Predict and compute error
y2_pred_phi = predict(X2_mapped, w_phi, b_phi)
error_phi = np.mean(y2_pred_phi != y2)

print(f"Misclassification error after phi transformation: {error_phi:.2f}")


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter the transformed data
ax.scatter(X2_mapped[y2 == 1, 0], X2_mapped[y2 == 1, 1], X2_mapped[y2 == 1, 2], color='blue', label='Class +1')
ax.scatter(X2_mapped[y2 == -1, 0], X2_mapped[y2 == -1, 1], X2_mapped[y2 == -1, 2], color='red', label='Class -1')

# Plot decision plane (only if w[2] ≠ 0)
if w_phi[2] != 0:
    x_plane, y_plane = np.meshgrid(
        np.linspace(np.min(X2_mapped[:, 0]), np.max(X2_mapped[:, 0]), 30),
        np.linspace(np.min(X2_mapped[:, 1]), np.max(X2_mapped[:, 1]), 30)
    )
    z_plane = -(w_phi[0] * x_plane + w_phi[1] * y_plane + b_phi) / w_phi[2]
    ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.3, color='gray')

ax.set_title(f"(g) SVM Decision Plane After phi Mapping\nError: {error_phi:.2f}")
ax.set_xlabel("1(x) = x₁")
ax.set_ylabel("2(x) = x₂")
ax.set_zlabel("3(x) = √(x₁² + x₂²)")
ax.legend()
plt.tight_layout()
plt.show()
