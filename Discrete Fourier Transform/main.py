import matplotlib.pyplot as plt
from math import pi as PI
import math
import cmath
import numpy as np


def dft (data):
    data_size = len(data)
    print("len data=", len(data))
    result = []
    if data_size:
        
        pi_div_N_mult_two = 2*PI/data_size #optimization
        for k in range (0, data_size):
            x_k_real = 0; #result
            x_k_im = 0; #result
            
            for n in range (0, data_size):
                x_n = data[n]
                arg = k*n*pi_div_N_mult_two
                x_k_real += x_n * math.cos(arg)
                x_k_im -= x_n * math.sin(arg)
            
            result.append(complex(x_k_real, x_k_im))
    
    return result
    
            
        

def square(data):
    result_data = []
    data_size = len(data)

    
    for i in range (0, data_size):
        val = data[i]*data[i]
        result_data.append(val)
    return result_data




"""x = np.linspace(-np.pi, np.pi, 10)
y=np.sin(x)

res = np.fft.fft(y)


my_res = dft(y)

for i in range (0, len(res)):
    print("your:", my_res[i], "np:", res[i])"""

def extract_real(complex_data):
    data_size = len(complex_data)
    result = []
    for i in range(0, data_size):
        real_val = complex_data[i].real
        result.append(real_val)
    return result

def extract_im(complex_data):
    data_size = len(complex_data)
    result = []
    for i in range(0, data_size):
        real_val = complex_data[i].imag
        result.append(real_val)
    return result

x = np.linspace(-np.pi, np.pi, 100)
y=np.sin(x)

fft_lib_res = np.fft.fft(y)
fft_lib_res_real = extract_real(fft_lib_res)

my_res = dft(y)
my_res_real = extract_real(my_res)


#fig = plt.figure()
#subplot_sin = fig.add_subplot(4, 4, 1)
#subplot_sin.set_title("sin(x)")
#subplot_sin.xaxis.set_ticks_position('bottom')
#subplot_sin.yaxis.set_ticks_position('left')

plt.subplot(441)
plt.plot(x, my_res_real, 'b')
plt.plot(x, fft_lib_res_real, 'r--')

plt.subplot(442)
plt.plot(x, y, 'r')


#plt.plot(x, y, 'b')
plt.show()


#data = [1, 2, 3, 4, 5]
#res = square(data)
"""print(res)

res = dft(data)
print(res)

plt.plot(res)
plt.ylabel("result")
plt.show()"""
        
