import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from math import pi as PI
import math
import cmath
import numpy as np
from scipy import signal
import random as rand

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

def get_spacing_period (left, right, numb_of_points):
    return (right-left)/numb_of_points

  
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
    subplot = fig.add_subplot(1, 3, 1)
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

    subplot = fig.add_subplot(1, 3, 2)
    subplot.set_title(title + " Fourier Image(Im)")
    subplot.xaxis.set_ticks_position('bottom')
    subplot.yaxis.set_ticks_position('left')
    
    subplot.plot(fftshift(xf), abs(fftshift(my_res_im))/numb_of_signal_pts, 'b', label = 'My dft, Im')
    subplot.plot(fftshift(xf), abs(fftshift(fft_lib_res_im))/numb_of_signal_pts, 'r--', label = 'Lib dft, Im')
    subplot.plot(fftshift(xf), abs(fftshift(diff_im))/numb_of_signal_pts, 'g', label = 'difference')
    subplot.legend()

    subplot = fig.add_subplot(1, 3, 3)
    subplot.set_title(title + "Original")
    subplot.xaxis.set_ticks_position('bottom')
    subplot.yaxis.set_ticks_position('left')
    subplot.plot(x, y, 'b', label = 'Original curve')
    subplot.legend()

    fig.show()





# number of signal points
N = 400

#sin(pi*x)
left_border = -2*np.pi
right_border = 2*np.pi
x = np.linspace(left_border, right_border, N)
y = np.sin(2*np.pi*x)
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'sin(2pi*x)')

#sin(2*pi*2x)
left_border = -2*np.pi
right_border = 2*np.pi
x = np.linspace(left_border, right_border, N)
y = np.sin(2*np.pi*2*x)
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'sin(2pi*x*2)')

#sin(2pi*2x) + sin(2pi*4x)
left_border = -2*np.pi
right_border = 2*np.pi
x = np.linspace(left_border, right_border, N)
y = np.sin(2*np.pi*2*x) + np.sin(2 * np.pi * 4*x)
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'sin(2pi*2x)+sin(2pi*4x)')

#exp(-x^2/2)
left_border = -50
right_border = 50
x = np.linspace(left_border, right_border, N)
y = np.exp(-0.5 * np.square(x))
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'exp(-x^2/2)')


#Dirac delta
left_border = -100
right_border = 100
x = np.linspace(left_border, right_border, N)
y = []
for i in range (0, len(x)):
    y.append(0)
y[round(len(y)/2)] = 100
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'Dirac delta')



#meander
meander_frequency = 5
left_border = 0
right_border = 1
x = np.linspace(0, 1, N)
y = signal.square(2 * np.pi * meander_frequency * x)
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'meander, f='+str(meander_frequency))

def spoil_data (data, spoil_intensity): #spoil intensity = 20 --> each 20th point will be spoiled, etc
    len_data = len(data)
    counter = 0
    for i in range (0, len_data):
        counter+=1
        if(counter == spoil_intensity):
            data[i] = rand.random()
            counter = 0
    return data

#blurred sin(2pi*2x) + sin(2pi*4x)
left_border = -2*np.pi
right_border = 2*np.pi
x = np.linspace(left_border, right_border, N)
y = np.sin(2*np.pi*2*x) + np.sin(2 * np.pi * 4*x)
y = spoil_data(y, 10)
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'SPOILED sin(2pi*2x)+sin(2pi*4x)')
