import matplotlib.pyplot as plt
import scipy.fft as sp 
from scipy.fft import fft, fftfreq, fftshift
from math import pi as PI
import math
import cmath
import numpy as np
from scipy import signal
import random as rand
from scipy.fftpack import fft, ifft



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

    def IFFT(self):
        ifft_y = sp.ifft(self.y)
        #spacing_period = (self.From-self.To)/self.N

        ifft_x = np.linspace(-1, 1, len(ifft_y))
        return [ifft_x, ifft_y]
        
        
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

# Скользящее окно
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

calc_and_plot(Window, sin_Func_a, 'noizy sin, w='+str(Omega))


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
    

def plot(x, y, title):
    fig = plt.figure()
    fig.suptitle(title)
    subplot = fig.add_subplot(111)
    
    subplot.plot(x, y, label='plot()')
    subplot.legend()
    fig.show()

def normalize(y, val):
    for i in range(0, len(y)):
        y[i] = y[i]/val
    return y    
    

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


x = np.linspace(Freq_character.From, Freq_character.To, len(res_y))


subplot = fig.add_subplot(122)
subplot.plot(x, calc_module(res_y), color = 'green', marker = '.', label = 'Cut Filter Charact')
#subplot.plot(Freq_character.x, Freq_character.y, 'r--', label='Source Filter Charact')

subplot.legend()
fig.show()






def calc_and_plot_filter(f_cut, filt_len, N_filt_points, src_Func, freq, title):
    # Через аналитическую функцию

    filter_from = -filt_len/2
    filter_to = filt_len/2
    
    x_filt = np.linspace(filter_from, filter_to, N_filt_points)

    # Импульсная характеристика нормированная
    h = 2*f_cut* np.sin(2*np.pi*f_cut*x_filt) / (2*np.pi*f_cut*x_filt) #np.sinc(2*f_cut * x_filt)
    h = normalize(h, abs(max(abs(h))))


    # Фильтрование
    y_filtered = np.convolve(src_Func.y, h)
    x_filtered_from = src_Func.From#min(src_Func.From, -filt_len/2)
    x_filtered_to = src_Func.To#max(src_Func.To, filt_len/2)
    x_filtered = np.linspace(x_filtered_from, x_filtered_to, len(y_filtered))
    
    fig = plt.figure()
    fig.suptitle(title)
    
    subplot = fig.add_subplot(133)
    subplot.plot(x_filt, h, label = 'Filter Impulse Characteristics')
    subplot.legend()

    subplot = fig.add_subplot(131)
    #subplot.set_xlim(min(x_filtered_from, src_Func.From), max(x_filtered_to, src_Func.To))
    subplot.plot(x_filtered, y_filtered, 'r--', label='Filtered Signal')
    subplot.legend()

    subplot = fig.add_subplot(132)
    subplot.plot(src_Func.x, src_Func.y, label = 'Source Signal')
    subplot.legend()

    fig.show()

    
    
    

# Через аналитическую функцию
f_cut = 0.1
f=2

filter_len = 50
N_points_filter = 1000

sin_Func = Function((FuncParams(-15/f, 15/f, 1, 1, 0, 1500)))
sin_Func.reset_x()

f1 = 3*f
f2 = 10*f
sin_Func.y = np.sin(f1 * 2*np.pi * sin_Func.x) + np.sin(f2 * 2*np.pi * sin_Func.x)
calc_and_plot_filter(f_cut, filter_len, N_points_filter, sin_Func, f, 'sin sum, f src='+str(f1)+' and '+ str(f2)+ ', f cut='+str(f_cut))


# Срезание частоты ещё пример
f_cut = 1
f=1
filter_len = 50
N_points_filter = 1000


noize_level=0.3
noize_intensity=0.20
sin_Func = Function((FuncParams(-np.pi, np.pi, 1, 1, 0, 500)))
sin_Func.reset_x()
sin_Func.y = np.sin(f * sin_Func.x)
sin_Func.noize(noize_intensity, noize_level)

f1 = 6
f2 = 20
calc_and_plot_filter(f_cut, filter_len, N_points_filter, sin_Func, f, 'sin sum, f src='+str(f1)+' and '+ str(f2)+ ', f cut='+str(f_cut))

# Пример как выше, но частота среза больше
f_cut = 1000
f = 50
filter_len = 50
N_points_filter = 1000
sin_Func = Function((FuncParams(-2/f, 2/f, 1, 1, 0, 1500)))
sin_Func.reset_x()

sin_Func.y = np.sin(f * 2 * np.pi * sin_Func.x) + 1*np.sin(3*f * 2 * np.pi * sin_Func.x) +1*np.sin(9*f * 2 * np.pi * sin_Func.x)
calc_and_plot_filter(f_cut, filter_len, N_points_filter, sin_Func, f, 'noizy sin, f cut ='+str(f_cut))

#
f_cut = 100
f = 50
filter_len = 50
N_points_filter = 1000
sin_Func = Function((FuncParams(-2/f, 2/f, 1, 1, 0, 1500)))
sin_Func.reset_x()

sin_Func.y = np.sin(f * 2 * np.pi * sin_Func.x) + 1*np.sin(3*f * 2 * np.pi * sin_Func.x) +1*np.sin(9*f * 2 * np.pi * sin_Func.x)
calc_and_plot_filter(f_cut, filter_len, N_points_filter, sin_Func, f, 'sin sum, f src=50, 150, 950, f cut ='+str(f_cut))

# Пример как выше, но частота среза больше
f_cut = 300
f = 50
filter_len = 50
N_points_filter = 1000
sin_Func = Function((FuncParams(-2/f, 2/f, 1, 1, 0, 1500)))
sin_Func.reset_x()

sin_Func.y = np.sin(f * 2 * np.pi * sin_Func.x) + 1*np.sin(3*f * 2 * np.pi * sin_Func.x) +1*np.sin(9*f * 2 * np.pi * sin_Func.x)
calc_and_plot_filter(f_cut, filter_len, N_points_filter, sin_Func, f, 'sin sum, f src=50, 150, 950, f cut ='+str(f_cut))



# Пример обрезания частоты
f_cut = 1
f=2
filter_len = 10
N_points_filter = 200
sin_Func = Function((FuncParams(-4/f, 4/f, 1, 1, 0, 1500)))
sin_Func.reset_x()

sin_Func.y = np.sin(20*f * 2 * np.pi * sin_Func.x)
calc_and_plot_filter(f_cut, filter_len, N_points_filter, sin_Func, f, 'f src=40, f cut=1')





# Пример сначала с обной частотой, потом с другой
N=500
f_1 = 1
f_2 = 3
left_border = -2*np.pi
right_border = 0
x = np.linspace(left_border, right_border, N)
y = np.sin(2*np.pi*f_1*x)

x = np.linspace(0, 2*np.pi, N)
y_temp = np.sin(2*np.pi*f_2*x)
y_concat=np.concatenate([y,y_temp])


Double_freq_func = Function((FuncParams(from_=-2*np.pi, to=2*np.pi, ampl=1, omega=1, shift=0, N=1000)))


x = np.linspace(-2*np.pi, 2*np.pi, 2*N)
Double_freq_func.x=x
Double_freq_func.y=np.concatenate([y,y_temp])

f_cut = 0.3 # А с f_cut = 1 почему-то не работает
f=2
filter_len = 50
N_points_filter = 600


calc_and_plot_filter(f_cut, filter_len, N_points_filter, Double_freq_func, f, 'Double freq, f src='+str(f_1)+' , ' + str(f_2)+'f_cut='+str(f_cut))

    
