import matplotlib.pyplot as plt
from math import pi as PI
import math
import cmath
import numpy as np


def dft (data):
    data_size = len(data)
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

def get_diff(list_a, list_b):
    result = []
    len_a = len(list_a)
    len_b = len(list_b)

    if len_a == len_b :
        for i in range (0, len_a):
            result.append(list_a[i] - list_b[i])
        return result


def calc_and_plot(x, y, title):
    fig = plt.figure()
        
    #library function calc
    fft_lib_res = np.fft.fft(y)
    fft_lib_res_real = extract_real(fft_lib_res)
    fft_lib_res_im = extract_im(fft_lib_res)

    #my_calc
    my_res = dft(y)
    my_res_real = extract_real(my_res)
    my_res_im = extract_im(my_res)

    #difference between my and library
    diff_re = get_diff(fft_lib_res_real, my_res_real)
    diff_im = get_diff(fft_lib_res_im, my_res_im)

    fig = plt.figure()
    subplot_sin = fig.add_subplot(1, 2, 1)
    subplot_sin.set_title(title + " Fourier Image(Re)")
    subplot_sin.xaxis.set_ticks_position('bottom')
    subplot_sin.yaxis.set_ticks_position('left')
    
    subplot_sin.plot(x, my_res_real, 'b', label = 'My dft, Re')
    subplot_sin.plot(x, fft_lib_res_real, 'r--', label = 'Lib dft, Re')
    subplot_sin.plot(x, diff_re, 'g', label = 'difference')
    subplot_sin.legend()

    subplot_sin = fig.add_subplot(1, 2, 2)
    subplot_sin.set_title(title + " Fourier Image(Im)")
    subplot_sin.xaxis.set_ticks_position('bottom')
    subplot_sin.yaxis.set_ticks_position('left')
    
    subplot_sin.plot(x, my_res_im, 'b', label = 'My dft, Im')
    subplot_sin.plot(x, fft_lib_res_im, 'r--', label = 'Lib dft, Im')
    subplot_sin.plot(x, diff_im, 'g', label = 'difference')
    subplot_sin.legend()

    fig.show()
    


x = np.linspace(-np.pi, np.pi, 100)
y=np.sin(x)
calc_and_plot(x, y, 'sin(x)')


x = np.linspace(-np.pi, np.pi, 100)
y=np.sin(2*x)
calc_and_plot(x, y, 'sin(2x)')

x = np.linspace(-2*np.pi, 2*np.pi, 100)
y=np.sin(10*x)
calc_and_plot(x, y, 'sin(10x)')
        
