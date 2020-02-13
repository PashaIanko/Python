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

def meander(x):
    len_ = len(x)
    for i in range(0, len_):
        if 1<=x[i]<=2:
            x[i]= 1
        else:
            x[i] = 0
    return x

def sin(x):
    for i in range(0, len(x)):
        x[i]=math.sin(x[i])
    return x

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

    def calc(self, func):
        self.reset_x()
        self.y = self.Ampl*func(self.W*(self.x+self.Shift))
        


def show_plot(Function_a, Function_b, convol_func, title):
    fig = plt.figure()
    fig.suptitle(title)
    subplot = fig.add_subplot(1,2,1)
    subplot.set_xlim(min(Function_a.From, Function_b.From), max(Function_a.To, Function_b.To))
    subplot.plot(Function_a.x, Function_a.y, 'b-')
    subplot.plot(Function_b.x, Function_b.y, 'g-')

    #subplot = fig.add_subplot(1, 2, 2)
    subplot.plot(convol_func.x, convol_func.y, 'r-')

    fig.show()
    return fig



def convolution(Function_a, Function_b):
    N = len(Function_a.x)
    M = len(Function_b.x)
    n_lim = N+M-2
    print(n_lim)
    a = Function_a.y
    b = Function_b.y
    res = []
    for n in range (0, n_lim):
        s_n = 0
        for m in range (0, n):
            elem = 0
            if 0<=m<N and 0<=n-m<M:               
                s_n += a[m]*b[n-m]
            
        res.append(s_n)
    res_func = Function((FuncParams(max(Function_a.From, Function_b.From), Function_a.To,0,0,0,len(res))))
    res_func.reset_x()
    res_func.y = res
    return res_func
        


        

meandr_Func_a = Function((FuncParams(0, 4, 1, 2, 0, 1000)))
meandr_Func_b = Function((FuncParams(0, 4, 1, 2, 0, 1000)))
meandr_Func_a.calc(meander)
meandr_Func_b.calc(meander)
res_func = convolution(meandr_Func_a, meandr_Func_b)
fig = show_plot(meandr_Func_a, meandr_Func_b, res_func, 'steps')


#sin
sin_Func_a = Function((FuncParams(0, np.pi*2, 1, 2, 3, 50)))
sin_Func_b = Function((FuncParams(0, np.pi*2, 1, 2, 0, 50)))
sin_Func_a.calc(np.sin)
sin_Func_b.calc(np.sin)
res_func = convolution(sin_Func_a, sin_Func_b)
fig = show_plot(sin_Func_a, sin_Func_b, res_func, 'sine')

#shifted sin
sin_Func_a = Function((FuncParams(-np.pi*2, np.pi*2, 1, 2, 3, 50)))
sin_Func_b = Function((FuncParams(-np.pi*2, np.pi*2, 1, 2, 0, 50)))
sin_Func_a.calc(np.sin)
sin_Func_b.calc(np.sin)
res_func = convolution(sin_Func_a, sin_Func_b)
fig = show_plot(sin_Func_a, sin_Func_b, res_func, 'sine')



