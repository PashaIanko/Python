import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


        
        
def func(t, y):
    return 2*(t*t + y)

def theory(x):
    return 1.5*np.exp(2*x)-x*x-x-0.5

t_from = 0
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
fig.show()





def func1(x, y):
    print("in func1")
    return 2*(x*x + y)

def func2(x, y):
    return 2*(x*x + y)

def func3(x, y):
    return 2*(x*x + y)

def func4(x, y):
    return 2*(x*x + y)


def calc_next(y, t, dt, func_arr):
    len_y = len(y)
    len_f_arr = len(func_arr)
    if len_y == len_f_arr:
        for i in range (0, len_y):
            y[i] = y[i]+dt*func_arr[i](t,y[i])
    
    return y

#vector function
func_arr = [func1, func2, func3, func4]

x_from = 0
x_to = 1
dx = 0.1

y1_start = 1
y2_start = 1
y3_start = 1
y4_start = 1

#Initial precondition for the equations system
y_start = [y1_start, y2_start, y3_start, y4_start]
x = x_from


y = y_start
#res_y = [y1_start, y2_start, y3_start, y4_start]
res_x = [x_from]
res_y = [[y1_start],[y2_start],[y3_start],[y4_start]]


while x<x_to:
    y = calc_next(y, x, dx, func_arr)

    for i in range (0, len(y)):
        res_y[i].append(y[i])
    
    x+=dx
    res_x.append(x)
    
print(res_y)


def theory(x):
    return 1.5*np.exp(2*x)-x*x-x-0.5


x_theory = np.linspace(x_from, x_to, 10)
y_theory_1 = 4*np.exp(-x) - np.exp(2*x)
y_theory_2 = np.exp(-x)-np.exp(2*x)
    
