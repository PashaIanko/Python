import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from math import pi as PI
import math
import cmath
import numpy as np
from scipy import signal

#my realization of dft
import dft as my_dft


   
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


"""def calc_and_plot(x, y, title):
    fig = plt.figure()
        
    #library function calc
    fft_lib_res = np.fft.fft(y)
    fft_lib_res_real = extract_real(fft_lib_res)
    fft_lib_res_im = extract_im(fft_lib_res)

    #my_calc
    my_res = my_dft.dft(y)
    my_res_real = extract_real(my_res)
    my_res_im = extract_im(my_res)

    #difference between my and library
    diff_re = get_diff(fft_lib_res_real, my_res_real)
    diff_im = get_diff(fft_lib_res_im, my_res_im)

    fig = plt.figure()
    subplot = fig.add_subplot(1, 2, 1)
    subplot.set_title(title + " Fourier Image(Re)")
    subplot.xaxis.set_ticks_position('bottom')
    subplot.yaxis.set_ticks_position('left')
    
    subplot.plot(x, my_res_real, 'b', label = 'My dft, Re')
    subplot.plot(x, fft_lib_res_real, 'r--', label = 'Lib dft, Re')
    subplot.plot(x, diff_re, 'g', label = 'difference')
    subplot.plot(x, y, 'yellow', label = 'Original')
    subplot.legend()

    subplot = fig.add_subplot(1, 2, 2)
    subplot.set_title(title + " Fourier Image(Im)")
    subplot.xaxis.set_ticks_position('bottom')
    subplot.yaxis.set_ticks_position('left')
    
    subplot.plot(x, my_res_im, 'b', label = 'My dft, Im')
    subplot.plot(x, fft_lib_res_im, 'r--', label = 'Lib dft, Im')
    subplot.plot(x, diff_im, 'g', label = 'difference')
    subplot.legend()

    fig.show()"""
    
def calc_and_plot(x, y, numb_of_signal_pts, spacing_period, title):
    fig = plt.figure()
        
    #library function calc
    fft_lib_res = np.fft.fft(y)
    fft_lib_res_real = extract_real(fft_lib_res)
    fft_lib_res_im = extract_im(fft_lib_res)

    #my_calc
    my_res = my_dft.dft(y)
    my_res_real = extract_real(my_res)
    my_res_im = extract_im(my_res)

    #difference between my and library
    diff_re = get_diff(fft_lib_res_real, my_res_real)
    diff_im = get_diff(fft_lib_res_im, my_res_im)

    fig = plt.figure()
    subplot = fig.add_subplot(1, 2, 1)
    subplot.set_title(title + " Fourier Image(Re)")
    subplot.xaxis.set_ticks_position('bottom')
    subplot.yaxis.set_ticks_position('left')

    #getting the sampling frequencies
    xf = fftfreq(numb_of_signal_pts, spacing_period)
        
    subplot.plot(fftshift(xf), abs(fftshift(my_res_real))/numb_of_signal_pts, 'b', label = 'My dft, Re')
    subplot.plot(fftshift(xf), abs(fftshift(fft_lib_res_real))/numb_of_signal_pts, 'r--', label = 'Lib dft, Re')
    subplot.plot(fftshift(xf), abs(fftshift(diff_re))/numb_of_signal_pts, 'g', label = 'difference')
    #subplot.plot(x, y, 'yellow', label = 'Original')
    subplot.legend()

    subplot = fig.add_subplot(1, 2, 2)
    subplot.set_title(title + " Fourier Image(Im)")
    subplot.xaxis.set_ticks_position('bottom')
    subplot.yaxis.set_ticks_position('left')
    
    subplot.plot(fftshift(xf), abs(fftshift(my_res_im))/numb_of_signal_pts, 'b', label = 'My dft, Im')
    subplot.plot(fftshift(xf), abs(fftshift(fft_lib_res_im))/numb_of_signal_pts, 'r--', label = 'Lib dft, Im')
    subplot.plot(fftshift(xf), abs(fftshift(diff_im))/numb_of_signal_pts, 'g', label = 'difference')
    subplot.legend()

    fig.show()

"""x = np.linspace(-10 * np.pi, 10 * np.pi, 200)
y = np.sin(x)
calc_and_plot(x, y, 'sin(x)')

x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.sin(10*x) + np.sin(5*x)
calc_and_plot(x, y, 'sin(10x)+sin(5x)')

#meander
meander_frequency = 5
x = np.linspace(0, 1, 1000, endpoint = True)
y = signal.square(2 * np.pi * meander_frequency * x)
calc_and_plot(x, y, 'meander, f='+str(meander_frequency))

#exp(-x^2/2)
x = np.linspace(-50, 50, 200)
y = np.exp(-0.5 * np.square(x))
calc_and_plot(x, y, 'exp(-x^2/2)')

        
#Diracks 
x = np.linspace(-100, 100, 200)
y = []
for i in range (0, len(x)):
    y.append(0)
y[round(len(y)/2)] = 100
calc_and_plot(x, y, 'Dirac delta')"""

def get_spacing_period (left, right, numb_of_points):
    return (right-left)/numb_of_points

# number of signal points
N = 400
# sample spacing
T = 1.0 / 800.0

left_border = -2*np.pi
right_border = 2*np.pi
x = np.linspace(left_border, right_border, N)
y = np.sin(2*np.pi*x)

calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'sin(2pi*x)')

#x = np.linspace(-2*np.pi, 2*np.pi, 100)
#y = np.sin(10*x) + np.sin(5*x)
#calc_and_plot(x, y, 'sin(10x)+sin(5x)')

#yf = my_dft.dft(y)
#xf = fftfreq(N, T)
#xf = fftshift(xf)
#yplot = fftshift(yf)



#plt.plot(xf, 1.0/N * np.abs(yplot))
#plt.grid()
#plt.show()
