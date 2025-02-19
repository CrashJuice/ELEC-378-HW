from scipy.io import loadmat
from scipy.io.wavfile import write
import numpy as np
import IPython

data1 = loadmat('cauchy_schwarz_decoding.mat')
data2 = loadmat('cauchy_schwarz_decoding_2.mat')

y = np.squeeze(data2['y2'])
c0 = np.squeeze(data1['c0'])
c1 = np.squeeze(data1['c1'])
chrp = np.squeeze(data2['chrp'])

print(chrp.size)
print(y.size)
print(c0.size)

conv_result = np.convolve(y, chrp[::-1], mode='valid')
thresh = 0.8 * np.max(conv_result)
indices = np.where(conv_result >= thresh)[0]

if len(indices) >= 2:
    start_idx, end_idx = indices[0]+len(chrp), indices[-1]
    y_extracted = y[start_idx:end_idx]
print(y_extracted.size)

C = np.column_stack((c0, c1))
num_bits = len(y_extracted) // len(c0)
Y = y_extracted.reshape(num_bits, len(c0))
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

# Audio portion:
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
