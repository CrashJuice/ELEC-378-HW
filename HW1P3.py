from scipy.io import loadmat
from scipy.io.wavfile import write
import numpy as np
import IPython
data = loadmat('cauchy_schwarz_decoding.mat')
y = data['y']
c0 = data['c0']
c1 = data['c1']
y = np.squeeze(y)
c0 = np.squeeze(c0)
c1 = np.squeeze(c1)
C = np.column_stack((c0, c1))
num_bits = len(y) // len(c0)
Y = y.reshape(num_bits, len(c0))
S = Y @ C 
bits = np.argmax(S, axis=1) 
strResult = ''.join(str(n) for n in bits)  
byteResult = [
    int(strResult[i : i + 8][::-1], 2)  
    for i in range(0, len(strResult), 8)
]
arrayResult = np.asarray(byteResult, dtype='uint8')
with open('decoded.jpg', 'wb') as f:
    f.write(arrayResult)
#Audio portion:
X = C[:, bits] 
x = X.flatten('F')  
y = y.flatten()
y = np.int16(y / np.max(np.abs(y)) * 32767) 
x = np.int16(x / np.max(np.abs(x)) * 32767)  
fs = 44100
write("y.wav", fs, y)
write("x.wav", fs, x)
IPython.display.Audio("y.wav") 
IPython.display.Audio("x.wav")  