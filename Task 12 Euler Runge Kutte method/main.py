import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


        
        
def func(t, y):
    return 2*(t*t + y)

def theory(x):
    return 1.5*np.exp(2*x)-x*x-x-0.5

'''t_from = 0
t_to = 1

y_start = 1 #y(t_from)

t=t_from
y=y_start
dt=0.1
res_y = [y_start]
res_t = [t_from]
while t<1:
    y = y + dt*func(t, y)
    res_y.append(y)
    t+=dt
    res_t.append(t)

x_theory = np.linspace(t_from, t_to, 10)
y_theory = theory(x_theory)

fig = plt.figure()
subplot = fig.add_subplot(111)
subplot.plot(res_t, res_y)
subplot.plot(x_theory, y_theory)
fig.show()'''


def Euler(y_init_cond, x_from, x_to, dx, functions_arr):
    x = x_from
    y = list(y_init_cond)
    res_x = [x_from]
    res_y = [[y_init_cond[0]],[y_init_cond[1]],[y_init_cond[2]],[y_init_cond[3]]]

    while x<x_to:
    #print(y)
        y = calc_next(y, x, dx, functions_arr)

        for i in range (0, len(y)):
            res_y[i].append(y[i])
        x+=dx
        res_x.append(x)
    return [res_x, res_y]

def calc_next_RK(y, x, dx, function_arr):
    #k1 = dx*function(x, y)
    #k2 = dx*function(x+dx/2,y+k1/2)
    #k3 = dx*function(x+dx/2,y+k2/2)
    #k4 = dx*function(x+dx,y+k3)
   # y_k1 = [y_+k1/2 for y_ in y]
   # y_k2 = [y_+k2/2 for y_ in y]
   # y_k3 = [y_+k3 for y_ in y]

    for i in range (0, len(y)):
        k1 = dx*function_arr[i](x, y)
        y_k1 = [y_+k1/2 for y_ in y]
        k2 = dx*function_arr[i](x+dx/2,y_k1)#y+k1/2)
        y_k2 = [y_+k2/2 for y_ in y]
        k3 = dx*function_arr[i](x+dx/2,y_k2)#y+k2/2)
        y_k3 = [y_+k3 for y_ in y]
        k4 = dx*function_arr[i](x+dx,y_k3)#y+k3)
        y[i] += (k1+2*(k2+k3)+k4)/6
        
    return y#+(k1+2*(k2+k3)+k4)/6

def RungeKutt(y_init_cond, x_from, x_to, dx, func_arr):
    x = x_from
    res_x = [x_from]
    res_y = [[y_init_cond[0]], [y_init_cond[1]], [y_init_cond[2]], [y_init_cond[3]]]
    y = list(y_init_cond)

    y_len = len(y)
    while x<x_to:
        
        y = calc_next_RK(y, x, dx, func_arr)

        for i in range(0, y_len):
            res_y[i].append(y[i])
        x+=dx
        res_x.append(x)

    return [res_x, res_y]


def func1(x, y):
    return -2*y[0]+4*y[1]

def func2(x, y):
    return -y[0]+3*y[1]

def func3(x, y):
    return -2*y[0]+4*y[1]

def func4(x, y):
    return -y[0]+3*y[1]


def calc_next(y, x, dx, func_arr):
    len_y = len(y)
    len_f_arr = len(func_arr)
    if len_y == len_f_arr:
        for i in range (0, len_y):
            y[i] = y[i]+dx*func_arr[i](x,y)
    
    return y

#vector function
func_arr = [func1, func2, func3, func4]

x_from = 0
x_to = 1
N = 50
dx = (x_to - x_from) / N

y1_start = 3
y2_start = 0
y3_start = 3
y4_start = 0

#Initial precondition for the equations system
y_start = [y1_start, y2_start, y3_start, y4_start]


#Calc Euler Method
[res_x, res_y] = Euler(y_start, x_from, x_to, dx, func_arr)

[res_x_RK, res_y_RK] = RungeKutt(y_start, x_from, x_to, dx, func_arr)

    
x_theory = np.linspace(x_from, x_to, N)
y_theory_1 = 4*np.exp(-x_theory) - np.exp(2*x_theory)
y_theory_2 = np.exp(-x_theory)-np.exp(2*x_theory)

#plotting
fig = plt.figure()
subplot = fig.add_subplot(111)

subplot.plot(x_theory, y_theory_1, 'b--', label='y1(x)theory')
subplot.plot(x_theory, y_theory_2, 'r--', label='y2(x)theory')
subplot.plot(res_x, res_y[0], 'b', label = 'y1(x) Euler')
subplot.plot(res_x, res_y[1], 'r', label = 'y2(x) Euler')
subplot.plot(res_x_RK, res_y_RK[0], 'black', label = 'y1(x) Runge Kutt')
subplot.plot(res_x_RK, res_y_RK[1], 'black', label = 'y2(x) Runge Kutt')
subplot.legend()
fig.show()
    
