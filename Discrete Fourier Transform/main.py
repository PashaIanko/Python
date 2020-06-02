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


def calc_module(c_res):
    res = []
    for i in c_res:
        module = (i.imag**2 + i.real**2)**0.5
        res.append(module)
    return res

def calc_phase(fft_res):
    fft = list(fft_res)
    phases=[]
    for i in range(0, len(fft)):
        phase = cmath.phase(fft[i])
        phases.append(phase)
    return phases

def calc_and_plot(x, y, numb_of_signal_pts, spacing_period, title):
    fig = plt.figure()
        
    #library function calc
    fft_lib_res = np.fft.fft(y)
    fft_lib_res_real = extract_real(fft_lib_res)
    fft_lib_res_im = extract_im(fft_lib_res)


    #
    phases = calc_phase(fft_lib_res)
    
    
    #my_calc
    my_res = my_dft.dft(y)
    my_res_real = extract_real(my_res)
    my_res_im = extract_im(my_res)

    #difference between my and library
    diff_re = get_diff(fft_lib_res_real, my_res_real)
    diff_im = get_diff(fft_lib_res_im, my_res_im)
    xf = fftfreq(numb_of_signal_pts, spacing_period)

    fig = plt.figure()
    ''' subplot = fig.add_subplot(1, 4, 1)
    subplot.set_title(title + " Fourier Image(Re)")
    subplot.xaxis.set_ticks_position('bottom')
    subplot.yaxis.set_ticks_position('left')'''

    '''#getting the sampling frequencies
    
    subplot.plot(fftshift(xf), abs(fftshift(my_res_real))/numb_of_signal_pts, 'b', label = 'My dft, Re')
    subplot.plot(fftshift(xf), abs(fftshift(fft_lib_res_real))/numb_of_signal_pts, 'r--', label = 'Lib dft, Re')
    subplot.plot(fftshift(xf), abs(fftshift(diff_re))/numb_of_signal_pts, 'g', label = 'difference')
    #subplot.plot(x, y, 'yellow', label = 'Original')
    subplot.legend()'''

    '''subplot = fig.add_subplot(1, 4, 2)
    subplot.set_title(title + " Fourier Image(Im)")
    subplot.xaxis.set_ticks_position('bottom')
    subplot.yaxis.set_ticks_position('left')
    
    subplot.plot(fftshift(xf), abs(fftshift(my_res_im))/numb_of_signal_pts, 'b', label = 'My dft, Im')
    subplot.plot(fftshift(xf), abs(fftshift(fft_lib_res_im))/numb_of_signal_pts, 'r--', label = 'Lib dft, Im')
    subplot.plot(fftshift(xf), abs(fftshift(diff_im))/numb_of_signal_pts, 'g', label = 'difference')
    subplot.legend()'''

    subplot = fig.add_subplot(1, 2, 1)
    subplot.set_title(title + "Original")
    subplot.xaxis.set_ticks_position('bottom')
    subplot.yaxis.set_ticks_position('left')
    subplot.plot(x, y, 'b', label = 'Original curve')
    subplot.legend()


    my_res_module = calc_module(my_res)
    lib_res_module = calc_module(fft_lib_res)
    diff_module = get_diff(my_res_module, lib_res_module)
    
    subplot = fig.add_subplot(1, 2, 2)
    subplot.set_title(title + "Fourier image module")
    subplot.xaxis.set_ticks_position('bottom')
    subplot.yaxis.set_ticks_position('left')
    subplot.set_xlim(0, max(fftshift(xf)))
    subplot.plot(fftshift(xf), abs(fftshift(my_res_module))/numb_of_signal_pts, 'b', label = 'My dft, |z|')
    subplot.plot(fftshift(xf), abs(fftshift(lib_res_module))/numb_of_signal_pts, 'r--', label = 'Lib dft, |z|')
    subplot.plot(fftshift(xf), abs(fftshift(diff_module))/numb_of_signal_pts, 'g', label = 'difference')
    subplot.plot(fftshift(xf), phases, 'black', label='phase')
    subplot.legend()

    fig.show()





#sin(2pi*2x) + sin(2pi*4x)
'''N=1000
left_border = -2*np.pi
right_border = 2*np.pi
x = np.linspace(left_border, right_border, N)
y = np.sin(np.multiply(2*np.pi*2, x)) + np.sin(np.multiply(2 * np.pi * 4, x))
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'sin(2pi*2x)+sin(2pi*4x)')'''

'''
#exp(-x^2/2)
N = 400
left_border = -50
right_border = 50
x = np.linspace(left_border, right_border, N)
y = np.exp(np.multiply(-0.5, np.square(x)))
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'exp(-x^2/2)')


#Dirac delta
N=400
left_border = -100
right_border = 100
x = np.linspace(left_border, right_border, N)
y = []
for i in range (0, len(x)):
    y.append(0)
y[round(len(y)/2)] = 100
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'Dirac delta')
'''


#############
#обратное фурье от сдвига
N=500
left_border = -2*np.pi
right_border = 2*np.pi
shift = np.pi
x = np.linspace(left_border, right_border, N)
y = np.sin(2*np.pi*x+np.pi)

fft_lib_res = np.fft.fft(y)
fft_lib_res_real = extract_real(fft_lib_res)
fft_lib_res_im = extract_im(fft_lib_res)
lib_res_module = calc_module(fft_lib_res)
#my_calc

my_res = my_dft.dft(y)

xf = fftfreq(N, get_spacing_period(left_border, right_border, N))

reverse_dft = np.fft.ifft(fft_lib_res)
reverse_dft_ = np.fft.ifft(lib_res_module)
reverse_dft_module = calc_module(reverse_dft)


fig = plt.figure()
title = 'shifted sin '
subplot = fig.add_subplot(1, 2, 1)
subplot.set_title(title + "Original curve")
subplot.xaxis.set_ticks_position('bottom')
subplot.yaxis.set_ticks_position('left')
subplot.plot(x, y, 'b', label = 'Original curve')
subplot.plot(x, reverse_dft_module, 'r--', label = 'F-1[F[f(x)](w)], от компл.чисел')
#subplot.plot(x, calc_module(reverse_dft_), 'm--', label = 'Обр преобр от модуля')
subplot.legend()

my_res_module = calc_module(my_res)

diff_module = get_diff(my_res_module, lib_res_module)

subplot = fig.add_subplot(1, 2, 2)
subplot.set_title(title + "Fourier image module")
subplot.xaxis.set_ticks_position('bottom')
subplot.yaxis.set_ticks_position('left')
subplot.set_xlim(0, max(fftshift(xf)))
subplot.plot(fftshift(xf), abs(fftshift(my_res_module))/N, 'b', label = 'My dft, |z|')
subplot.plot(fftshift(xf), abs(fftshift(lib_res_module))/N, 'r--', label = 'Lib dft, |z|')
subplot.plot(fftshift(xf), abs(fftshift(diff_module))/N, 'g', label = 'difference')
subplot.legend()

fig.show()




'''#Double freq (begins with low, ends with high)
N=1000
left_border = -2*np.pi
right_border = 0
x = np.linspace(left_border, right_border, N)
y = np.sin(2*np.pi*2*x)
print(len(y))

x = np.linspace(0, 2*np.pi, N)
y_temp = np.sin(2*np.pi*6*x)
print(y_temp)
y_concat=np.concatenate([y,y_temp])
print("y_tmp, Y", len(y_temp), len(y))
x = np.linspace(-2*np.pi, 2*np.pi, 2*N)

calc_and_plot(x, np.concatenate([y, y_temp]), 2*N, get_spacing_period(left_border, right_border, 2*N), 'sin(2pi*4x) and sin(2pi*12x)')
'''

#meander
'''meander_frequency = 5
left_border = 0
right_border = 1
x = np.linspace(0, 1, N)
y = signal.square(2 * np.pi * meander_frequency * x)
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'meander, f='+str(meander_frequency))
'''



def spoil_data (data, spoil_intensity): #spoil intensity = 20 --> each 20th point will be spoiled, etc
    len_data = len(data)
    counter = 0
    for i in range (0, len_data):
        counter+=1
        if(counter == spoil_intensity):
            data[i] = rand.random()
            counter = 0
    return data

'''#blurred sin(2pi*2x) + sin(2pi*4x)
N=400
left_border = -2*np.pi
right_border = 2*np.pi
x = np.linspace(left_border, right_border, N)
y = np.sin(2*np.pi*2*x) + np.sin(2 * np.pi * 4*x)
y = spoil_data(y, 10)
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'SPOILED sin(2pi*2x)+sin(2pi*4x)')


#Diff. amplitudes 10*sin(2pi*2x) + 3*sin(2pi*4x)
left_border = -2*np.pi
right_border = 2*np.pi
x = np.linspace(left_border, right_border, N)
y = 10*np.sin(2*np.pi*2*x) + 3*np.sin(2 * np.pi * 4*x)
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), '10*sin(2pi*2x)+3*sin(2pi*4x)')
'''


def get_step(x, from_, to_):
    y = []
    for x_ in x:
        if from_ <= x_ <= to_:
            y.append(1)
        else:
            y.append(0)
    return y

'''#step
N=300
left_border = -2*np.pi
right_border = 2*np.pi
x = np.linspace(left_border, right_border, N)
y = get_step(x, -1, 1)
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'Step')

#blurred step
N=300
left_border = -2*np.pi
right_border = 2*np.pi
x = np.linspace(left_border, right_border, N)
y = get_step(x, -1, 1)
y=spoil_data(y, 5)
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'Spoiled Step')



# Малая выборка
N=30
left_border = -2*np.pi
right_border = 2*np.pi
x = np.linspace(left_border, right_border, N)
y = get_step(x, -1, 1)
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'N=30 малая выборка')


#step wide
N=400
left_border = -2*np.pi
right_border = 2*np.pi
x = np.linspace(left_border, right_border, N)
y = get_step(x, -3, 3)
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'Step')'''


'''#sin, low frequency
left_border = -2*np.pi
right_border = 2*np.pi
x = np.linspace(left_border, right_border, N)
freq = 0.2
y = np.sin(2*np.pi*freq*x)
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'sin(2pi*'+str(freq)+'x+100)')
'''

'''#couple of sin, close freq
N=600
left_border = -3*np.pi
right_border = 3*np.pi
x = np.linspace(left_border, right_border, N)
freq = 2
y = np.sin(2*np.pi*freq*x) + np.sin(2*np.pi*(freq+0.2)*x)
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'close freq sin(2pi*2x), close(2pi*2.2x)')


#far freq of sin
N=600
left_border = -2*np.pi
right_border = 2*np.pi
x = np.linspace(left_border, right_border, N)
freq = 1
freq_ = 5

y = np.sin(x) + np.sin(3*x)+np.sin(5*x)
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'sin(x)+sin(3x)+sin(5x)')


#freq more than Nykvist freq
N=7 # При использовании 700 отсчётов, пик на 4 Гц, всё корректно. Но, когда
# рассмотрим критическое N такое что частота > N/2 --> неверный спектр, информации о частоте 4 Гц нет. --> При N=7, действительно получили плохой спектр

left_border = -np.pi
right_border = np.pi
x = np.linspace(left_border, right_border, N)
y = np.sin(2*np.pi*4*x) 
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'More than Nykvist')'''



#############################
# Исследование спектра для двух частот
#Double freq (begins with low, ends with high)
N=500
left_border = -2*np.pi
right_border = 0
x = np.linspace(left_border, right_border, N)
y = np.sin(2*np.pi*0.5*x)
print(len(y))

x = np.linspace(0, 2*np.pi, N)
y_temp = np.sin(2*np.pi*x)
y_concat=np.concatenate([y,y_temp])
print("y_tmp, Y", len(y_temp), len(y))
x = np.linspace(-2*np.pi, 2*np.pi, 2*N)

calc_and_plot(x, np.concatenate([y, y_temp]), 2*N, get_spacing_period(left_border, right_border, 2*N), 'sin(2pi*4x) and sin(2pi*12x)')

N=1000
left_border = -2*np.pi
right_border = 2*np.pi
x = np.linspace(left_border, right_border, N)
y = np.sin(2*np.pi*x) + np.sin(2*np.pi*2*x)
calc_and_plot(x, y, N, get_spacing_period(left_border, right_border, N), 'sin(2pi*x)+sin(2pi*2x)')



