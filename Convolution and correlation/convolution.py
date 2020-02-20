import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from math import pi as PI
import math
import cmath
import numpy as np
from scipy import signal
import random as rand
import tkinter as tk
import time



from tkinter import*

def meander(x, *args, **kwargs):
    len_ = len(x)
    from_, to_ = args
      
    
    for i in range(0, len_):
        if from_<=x[i]<=to_:
            x[i]= 1
        else:
            x[i] = 0
    
    return x

def triangle(x, *args, **kwargs):
    len_ = len(x)
    from_, to_ = args
        
    
    b = -to_/(from_-to_)
    a = 1/(from_-to_)
    
    for i in range(1, len_):
        if from_<=x[i]<=to_:
            x[i] = a*x[i]+b
        else:
            x[i]=0
    return x

    

def sin(x):
    for i in range(0, len(x)):
        x[i]=math.sin(x[i])
    return x

def calc_and_plot(func_a, func_b, title):
    [conv_func, corr_func] = calc_and_normalize(func_a, func_b)
    [conv_func_reverse, corr_func_reverse] = calc_and_normalize(func_b, func_a)
    
    lib_conv_array = np.convolve(func_b.y, func_a.y)
    lib_corr_array = signal.correlate(func_b.y, func_a.y)
       
    lib_conv_func = Function(FuncParams(min(func_a.From, func_b.From), max(func_a.To, func_b.To), 0,0,0, len(lib_conv_array)))
    lib_conv_func.y = lib_conv_array
    lib_conv_func.reset_x()
    lib_conv_func.normalize(max(lib_conv_func.y))

    lib_corr_func = Function(FuncParams(min(func_a.From, func_b.From), max(func_a.To, func_b.To), 0,0,0, len(lib_corr_array)))
    lib_corr_func.y = lib_corr_array
    lib_corr_func.reset_x()
    lib_corr_func.normalize(max(lib_corr_func.y))
    
    fig = show_plot(func_a, func_b, conv_func, corr_func, conv_func_reverse, corr_func_reverse, title, lib_conv_func, lib_corr_func)


class FuncParams:
    def __init__(self, from_, to, ampl, omega, shift, N):
        self.From = from_
        self.To = to
        self.Ampl = ampl
        self.Omega = omega
        self.shift = shift
        self.N = N

class Function :
    x = []
    y = []
    
    def __init__(self, FuncParams):
      
       self.From = FuncParams.From
       self.To = FuncParams.To
       self.W = FuncParams.Omega
       self.N = FuncParams.N
       self.Shift = FuncParams.shift
       self.Ampl = FuncParams.Ampl

    def reset_x(self):
        self.x.clear()
        self.x = np.linspace(self.From, self.To, self.N)

    def calc(self, func, *args, **func_args):
        self.reset_x()
        for key, val in func_args.items():
            print(key, val)
        
        self.y = self.Ampl*func(self.W*(self.x+self.Shift), *args, **func_args)
    def plot(self):
        fig = plt.figure()
        subplot = fig.add_subplot(111)
        subplot.plot(self.x, self.y, 'b')
        fig.show()

    def normalize(self, value):
        len_ = len(self.y)
        for i in range (0, len_):
            self.y[i] = self.y[i]/value

    def noize(self, intensity, level):
        if intensity >=0:
            increment = 1/intensity
            len_ = len(self.y)
            x = 0
            for i in range(0, len_):
                x+=increment
                if(x>=1):
                    x=0
                    self.y[i] += rand.uniform(-level, +level)

    def reverse(self):
        # Вернуть вместо f(t) --> f(-t)
        x_ = list(self.x)
        for i in range (0, len(x_)):
            x_[i] = -x_[i]
        
        x_ = np.sort(x_)
        y_ = list(np.flip(self.y))
        res = Function(FuncParams(self.From, self.To, self.Ampl, self.W, self.Shift, self.N))
        res.x = x_
        res.y = y_
        return res
        
def get_diff(my_res_y, lib_res_y):
    print('type=', type(lib_res_y), type(my_res_y))
    lib_res = list(lib_res_y)
    diff = []
    for i in range(0, min(len(my_res_y), len(lib_res_y))):
        diff.append(lib_res[i]-my_res_y[i])
        
    return diff

def show_plot(Function_a, Function_b, convol_func, corr_func, conv_func_reverse, corr_func_reverse, title, lib_conv_func, lib_corr_func):
    fig = plt.figure()
    fig.suptitle(title)
    subplot = fig.add_subplot(1,2,1)
    subplot.set_title('FuncA*FuncB Convolution')
    subplot.set_xlim(min(Function_a.From, Function_b.From), max(Function_a.To, Function_b.To))
    subplot.plot(Function_a.x, Function_a.y, 'b', label='func A')
    subplot.plot(Function_b.x, Function_b.y, 'r--', label='func B')

    subplot.plot(convol_func.x, convol_func.y, 'g-', label='Convolution')
    subplot.plot(lib_conv_func.x, lib_conv_func.y, 'k--', label='Library Convolution')

    diff_y = get_diff(convol_func.y, lib_conv_func.y)
    subplot.plot(convol_func.x, diff_y, 'yellow', label='Diff with lib')
    subplot.legend()
   
    

    subplot = fig.add_subplot(1, 2, 2)
    subplot.set_title('FuncA*FuncB Correlation')
    subplot.set_xlim(min(Function_a.From, Function_b.From), max(Function_a.To, Function_b.To))
    subplot.plot(Function_a.x, Function_a.y, 'b', label='func A')
    subplot.plot(Function_b.x, Function_b.y, 'r--', label='func B')
    
    subplot.plot(corr_func.x, corr_func.y, 'g-', label='Correlation')
    subplot.plot(lib_corr_func.x, lib_corr_func.y, 'k--', label='Library Correlation')

    print(len(corr_func.y), len(lib_corr_func.y))
    diff_y = get_diff(corr_func.y, lib_corr_func.y)
    subplot.plot(corr_func.x, diff_y, 'yellow', label = 'Diff with lib')
    
    subplot.legend()

    ######
    #plotting for reverse correlation and convolution
    #subplot.set_title('FuncB*FuncA')
    #subplot.set_xlim(min(Function_a.From, Function_b.From), max(Function_a.To, Function_b.To))
    #subplot.plot(Function_a.x, Function_a.y, 'b-', label='func A')
    #subplot.plot(Function_b.x, Function_b.y, 'g-', label='func B')
    #subplot.plot(conv_func_reverse.x, conv_func_reverse.y, 'r-', label='Conv reverse')
    #subplot.plot(corr_func_reverse.x, corr_func_reverse.y, 'm--', label='Corr reverse')
    #subplot.legend()
    #fig.legend()
    fig.show()
    return fig



def convolution(Function_a, Function_b):
    N = len(Function_a.y)#(Function_a.x)
    M = len(Function_b.y)#(Function_b.x)
    n_lim = N+M-2
    a = Function_a.y
    b = Function_b.y
    res = []
    for n in range (0, n_lim):
        s_n = 0
        for m in range (0, N):#n):
            elem = 0
            if 0<=m<N and 0<=n-m<M:               
                s_n += a[m]*b[n-m]
            
        res.append(s_n)
    res_func = Function((FuncParams(min(Function_a.From, Function_b.From), max(Function_a.To, Function_b.To),0,0,0,len(res))))
    res_func.reset_x()
    res_func.y = res
    return res_func

def correlation(Func_a, Func_b):
    N = len(Func_a.y)#(Function_a.x)
    M = len(Func_b.y)#(Function_b.x)
    #F(t) cross-cor G(T) == F*(-t) convolute G(t)
    Func_a_reverse = Func_a.reverse()
    return convolution(Func_b, Func_a_reverse)
        
def calc_and_normalize(func_a, func_b):
    conv_func = convolution(func_a, func_b)
    corr_func = correlation(func_a, func_b)

    conv_func.normalize(max(conv_func.y))
    corr_func.normalize(max(corr_func.y))
    return [conv_func, corr_func]

        
'''#rectangle
meandr_Func_a = Function((FuncParams(from_=0, to=4, ampl=10, omega=1, shift=0, N=100)))
meandr_Func_b = Function((FuncParams(0, 4, 10, 1, 0, 100)))
meandr_Func_a.calc(meander, 1, 2)
meandr_Func_b.calc(meander, 1, 2)
res_func = convolution(meandr_Func_a, meandr_Func_b)
res_func.normalize(max(res_func.y))
fig = show_plot(meandr_Func_a, meandr_Func_b, res_func, 'identic meander')'''

'''#shifted rectangle
meandr_Func_a = Function((FuncParams(from_=-1, to=4, ampl=1, omega=1, shift=0, N=100)))
meandr_Func_b = Function((FuncParams(from_=-1, to=4, ampl=1, omega=1, shift=0, N=100)))
meandr_Func_a.calc(meander, 0, 2)
meandr_Func_b.calc(meander, 1, 3)
res_func = convolution(meandr_Func_a, meandr_Func_b)
res_func.normalize(max(res_func.y))
fig = show_plot(meandr_Func_a, meandr_Func_b, res_func, 'shifted meander')'''


#sin identic
sin_Func_a = Function((FuncParams(0, np.pi*2, 1, 3, 0, 50)))
sin_Func_b = Function((FuncParams(0, np.pi*2, 1, 3, 0, 50)))
sin_Func_a.calc(np.sin)
sin_Func_b.calc(np.sin)
calc_and_plot(sin_Func_a, sin_Func_b, 'identic sin')

#shifted sin
sin_Func_a = Function((FuncParams(-np.pi*2, np.pi*2, 1, 1, 1, 50)))
sin_Func_b = Function((FuncParams(-np.pi*2, np.pi*2, 1, 1, 0, 50)))
sin_Func_a.calc(np.sin)
sin_Func_b.calc(np.sin)
calc_and_plot(sin_Func_a, sin_Func_b, 'shifted sin')

#sin counter phase
sin_Func_a = Function((FuncParams(-np.pi*4, np.pi*4, 1, 1, np.pi, 50)))
sin_Func_b = Function((FuncParams(-np.pi*4, np.pi*4, 1, 1, 0, 50)))
sin_Func_a.calc(np.sin)
sin_Func_b.calc(np.sin)
calc_and_plot(sin_Func_a, sin_Func_b, 'sin conter phase')

#sin high freq + low freq
sin_Func_a = Function((FuncParams(-np.pi*2, np.pi*2, 1, 5, 1, 500)))
sin_Func_b = Function((FuncParams(-np.pi*2, np.pi*2, 1, 1, 0, 500)))
sin_Func_a.calc(np.sin)
sin_Func_b.calc(np.sin)

calc_and_plot(sin_Func_a, sin_Func_b, 'sin high + low freq')

#meander, opposed phase
meandr_Func_a = Function((FuncParams(from_=0, to=2, ampl=1, omega=1, shift=0, N=100)))
meandr_Func_b = Function((FuncParams(from_=0, to=2, ampl=1, omega=1, shift=0, N=100)))
meandr_Func_a.calc(meander, 1, 2)
meandr_Func_b.calc(meander, 0, 1)

calc_and_plot(meandr_Func_a, meandr_Func_b, 'meanders opposed phase')

#two noizy meanders
meandr_Func_a = Function((FuncParams(from_=0, to=3, ampl=1, omega=1, shift=0, N=100)))
meandr_Func_b = Function((FuncParams(from_=0, to=3, ampl=1, omega=1, shift=0, N=100)))
meandr_Func_a.calc(meander, 1, 2)
meandr_Func_b.calc(meander, 1, 2)
noize_level = 0.3
meandr_Func_a.noize(0.5, noize_level)
meandr_Func_b.noize(0.5, noize_level)

calc_and_plot(meandr_Func_a, meandr_Func_b, 'two noizy meanders')

#noizy sin
sin_Func_a = Function((FuncParams(-np.pi*2, np.pi*2, 1, 3, 0, 100)))
sin_Func_b = Function((FuncParams(-np.pi*2, np.pi*2, 1, 3, 0, 100)))
sin_Func_a.calc(np.sin)
sin_Func_b.calc(np.sin)
sin_Func_a.noize(0.5, 1)
sin_Func_b.noize(0.5, 1)
calc_and_plot(sin_Func_a, sin_Func_b, 'noizy sin')



#step + triangle
meandr_Func = Function((FuncParams(from_=0, to=3, ampl=1, omega=1, shift=0, N=100)))
meandr_Func.calc(meander, 1, 2)
triangle_Func = Function((FuncParams(from_=0, to=3, ampl=1, omega=1, shift=0, N=100)))
triangle_Func.calc(triangle, 1, 2)
calc_and_plot(meandr_Func, triangle_Func, 'step + triangle')


#Double freq (begins with low, ends with high)
N=500
left_border = -2*np.pi
right_border = 0
x = np.linspace(left_border, right_border, N)
y = np.sin(2*np.pi*x)
print(len(y))

x = np.linspace(0, 2*np.pi, N)
y_temp = np.sin(2*np.pi*3*x)
print(y_temp)
y_concat=np.concatenate([y,y_temp])
print("y_tmp, Y", len(y_temp), len(y))

Double_freq_func = Function((FuncParams(from_=-2*np.pi, to=2*np.pi, ampl=1, omega=1, shift=0, N=1000)))
Double_freq_func_ = Function((FuncParams(from_=-2*np.pi, to=2*np.pi, ampl=1, omega=1, shift=0, N=1000)))

x = np.linspace(-2*np.pi, 2*np.pi, 2*N)
Double_freq_func.x=x
Double_freq_func.y=np.concatenate([y,y_temp])

Double_freq_func_.x =x
Double_freq_func_.y=np.concatenate([y,y_temp])

calc_and_plot(Double_freq_func, Double_freq_func_, 'double freq')



'''#Через Фурье
sin_Func_a = Function((FuncParams(-2*np.pi, np.pi*2, 1, 3, 0, 50)))
sin_Func_b = Function((FuncParams(-2*np.pi, np.pi*2, 1, 3, 0, 50)))
sin_Func_a.calc(np.sin)
sin_Func_b.calc(np.sin)


[conv_func, corr_func] = calc_and_normalize(sin_Func_a, sin_Func_b)
[conv_func_reverse, corr_func_reverse] = calc_and_normalize(sin_Func_b, sin_Func_a)

y = fft(sin_Func_a.y)*np.conj(fft(sin_Func_b.y))
y_ = np.fft.ifft(y)
x = np.linspace(0,1, len(y_))
fig = plt.figure()
subplot = fig.add_subplot(111)
subplot.plot(x, y_)
fig.show()'''

