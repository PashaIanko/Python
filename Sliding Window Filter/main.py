import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from math import pi as PI
import math
import cmath
import numpy as np
from scipy import signal
import random as rand



def step(x, *args, **kwargs):
    step_from_, step_to_ = args # откуда до куда будут ненулевые значения
                                               # + область определения функции
    print(step_from_, step_to_)
    samples_num = len(x)
    print(samples_num)
    numb_of_pts_within = 0
    
    for x_ in x:
        if step_from_<=x_<=step_to_:
            numb_of_pts_within+=1 # колво точек попадающих в область ступеньки
    
    step_height = 1/numb_of_pts_within
    y = []
    for i in range(0, len(x)):
        if step_from_<=x[i]<=step_to_:
            y.append(step_height)
        else:
            y.append(0)
        
    return y

def meander(x, *args, **kwargs):
    len_ = len(x)
    from_, to_ = args
      
    
    for i in range(0, len_):
        if from_<=x[i]<=to_:
            x[i]= 1
        else:
            x[i] = 0
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
       # self.x.clear()
         self.x = np.linspace(self.From, self.To, self.N)

    def calc(self, func, *args, **func_args):
        self.reset_x()
        #for key, val in func_args.items():
         #   print(key, val)
        
        self.y = self.Ampl*func(self.W*(self.x+self.Shift), *args, **func_args)
    def plot(self):
        fig = plt.figure()
        subplot = fig.add_subplot(111)
        subplot.plot(self.x, self.y, 'b', marker='.')
        
        
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

    def plot_FFT(self):
         # Частотная характеристика окна (Фильтра)
        fft_lib_res = np.fft.fft(self.y)
        spacing_period = (self.From-self.To)/self.N

        xf = fftfreq(self.N, spacing_period)
        lib_res_module = calc_module(fft_lib_res)

        fig = plt.figure()
        subplot = fig.add_subplot(111)
        subplot.set_xlim(0, max(xf))
        subplot.plot(fftshift(xf), abs(fftshift(lib_res_module))/self.N, label='Impulse characteristics')
        subplot.legend()
        fig.show()

    def FFT(self):
        fft_lib_res = np.fft.fft(self.y)
        spacing_period = (self.From-self.To)/self.N

        xf = fftfreq(self.N, spacing_period)
        return [fftshift(xf), fftshift(fft_lib_res)]
        
        
def calc_module(c_res):
    res = []
    for i in c_res:
        module = (i.imag**2 + i.real**2)**0.5
        res.append(module)
    return res

def calc_and_plot(Filter_Func, Sample_Func, title):
    # Результат фильтра (Свёртка)
    conv = np.convolve(Filter_Func.y, Sample_Func.y)
    x = np.linspace(Sample_Func.From, Sample_Func.To, len(conv))
    #plt.plot(x, conv, color='r')
    
    # Частотная характеристика окна (Фильтра)
    fft_lib_res = np.fft.fft(Filter_Func.y)
    spacing_period = (Filter_Func.From-Filter_Func.To)/Filter_Func.N

    xf = fftfreq(Filter_Func.N, spacing_period)
    lib_res_module = calc_module(fft_lib_res)


    fig = plt.figure()
    fig.suptitle(title)

    # Результат сглаживания 
    subplot = fig.add_subplot(131)
    subplot.plot(Sample_Func.x, Sample_Func.y, color='b', label='Source Signal')
    subplot.plot(x, conv, color='r', label='Smoothened Signal')
    subplot.legend()

    # Фильтр
    subplot = fig.add_subplot(132)
    subplot.plot(Filter_Func.x, Filter_Func.y, color='k', label='Filter Func h(x)', marker='.')
    subplot.legend()

    # Фурье
    subplot = fig.add_subplot(133)
    #subplot.set_xlim(0, max(xf))
    subplot.plot(fftshift(xf), abs(fftshift(lib_res_module))/Filter_Func.N, label='Impulse characteristics')
    subplot.legend()
    
    fig.show()

'''# Скользящее окно
Window = Function(FuncParams(0, 3, 1, 1, 0, 100))
Window.calc(step, 1, 2)


# Шумная ступенька
meandr = Function((FuncParams(from_=0, to=3, ampl=1, omega=1, shift=0, N=500)))
meandr.calc(meander, 1, 2)
noize_level = 0.3
meandr.noize(0.2, noize_level)

calc_and_plot(Window, meandr, 'noizy step')


# Noizy sin
Omega = 2
noize_level=0.2
noize_intensity=0.15
sin_Func_a = Function((FuncParams(-np.pi*2, np.pi*2, 1, Omega, 0, 500)))
sin_Func_a.calc(np.sin)
sin_Func_a.noize(noize_intensity, noize_level)

calc_and_plot(Window, sin_Func_a, 'noizy sin, w='+str(Omega))'''


def cut_filter(x_fft, y_fft, cut_percent):
    res_y = []
    res_x = []

    middle_idx = int(len(x_fft)/2)
    max_index = int(len(y_fft)*cut_percent / 2)
    
    #print(type(x_fft), type(y_fft))
    #print('lens=', len(x_fft), len(y_fft))
    res_y = y_fft.tolist()[middle_idx - max_index : middle_idx+max_index]
    res_x = x_fft.tolist()[middle_idx - max_index : middle_idx + max_index]
    return [res_x, res_y]
    
    
    

# Проверка на частотную характеристику ограниченного фильтра
Filter_len_cut = 0.5 # процент, насколько усекаем множество значений импульсной характеристики (выбор длины фильтра)
Freq_character = Function((FuncParams(from_=0, to=400, ampl=1, omega=1, shift=0, N=1000)))
Freq_character.calc(meander, 0, 150)
#Freq_character.plot()


[x_fft, y_fft] = Freq_character.FFT()

fig = plt.figure()
subplot = fig.add_subplot(121)
subplot.plot(x_fft, calc_module(y_fft), label='Impulse characteristics')


[cut_fft_x, cut_fft_y] = cut_filter(x_fft, y_fft, Filter_len_cut) # срезаем часть фурье образа (выбор длины фильтра)
subplot.plot(cut_fft_x, calc_module(cut_fft_y), label='Cut Impulse characteristics', color = 'r', marker='.')

subplot.legend()


# После обратного Фурье от урезанной характеристики фильтра
res_y = np.fft.ifft(cut_fft_y)
#res_y = calc_module(res_y)

x = np.linspace(Freq_character.From, Freq_character.To, len(res_y))


subplot = fig.add_subplot(122)
subplot.plot(x, calc_module(res_y), color = 'green', marker = '.', label = 'Cut Filter Charact')
subplot.plot(Freq_character.x, Freq_character.y, 'r--', label='Source Filter Charact')

subplot.legend()
fig.show()


# Последний шаг - свертка сигнала c импульсной характеристикой
Omega = 3
sin_Func = Function((FuncParams(-np.pi*2, np.pi*2, 1, Omega, 0, 1000)))
sin_Func.calc(np.sin)
y = np.convolve(sin_Func.y, res_y)
y = calc_module(y)
x = np.linspace(-np.pi*2,np.pi*2, len(y))

fig = plt.figure()
subplot = fig.add_subplot(111)

subplot.plot(sin_Func.x, sin_Func.y, label='Source Signal')
subplot.plot(x, y, label='Filtered Signal', marker='.')
subplot.legend()
fig.show()







    
