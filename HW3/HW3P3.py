import numpy as np
mu = 0.01 
# initial guess 
w_gd = np.array([1.0, 1.0])
# gradient descent :3
for i in range(10000):
    w1, w2 = w_gd 
    grad_w1 = 8*w1 + w2
    grad_w2 = w1 + 8*w2
    gradient = np.array([grad_w1, grad_w2]) # gradient vector
    # update weights
    w_new = w_gd - mu * gradient
    w_gd = w_new
# stochastic gradient descent
w_sgd = np.array([1.0, 1.0]) 

choices = {
    'L1': np.array([[4, 1], [1, -8]]),
    'L2': np.array([[6, 4], [4, 10]]),
    'L3': np.array([[-2, -4], [-4, 6]])
}
for i in range(10000):
    w1, w2 = w_sgd
    choice = np.random.choice(['L1', 'L2', 'L3'])
    grad_matrix = choices[choice]
    grad = grad_matrix @ np.array([w1, w2])
    w_sgd -= mu * grad #replace old values with calulated 
print("Gradient Descent:", w_gd)
print("Stochastic Gradient Descent:", w_sgd)
