import numpy as np
import matplotlib.pyplot as plt

D=100
volume = np.squeeze(np.zeros((1,D)))




N = 100000

for d in np.arange(1, D):
    
    X = np.random.uniform(-1, 1, size=(N, d)) 

  
    l2_norms = np.linalg.norm(X, axis=1)

    
    outside_l2_ball = np.sum(l2_norms > 1)

    
    volume[d] = outside_l2_ball / N  

plt.plot(volume, marker='o')
plt.xlabel('Dimension (d)')
plt.ylabel('Fraction of L-infinity volume outside L-2 ball')
plt.title('Volume Difference Between L-infinity and L-2 Balls')
plt.grid()
plt.show()