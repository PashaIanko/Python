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
N = 100
dx = (x_to - x_from) / N

y1_start = 3
y2_start = 0
y3_start = 3
y4_start = 0

#Initial precondition for the equations system
y_start = [y1_start, y2_start, y3_start, y4_start]
x = x_from

y = y_start
res_x = [x_from]
res_y = [[y1_start],[y2_start],[y3_start],[y4_start]]


while x<x_to:
    print(y)
    y = calc_next(y, x, dx, func_arr)

    for i in range (0, len(y)):
        res_y[i].append(y[i])
    
    x+=dx
    res_x.append(x)
    
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
#subplot.plot(res_x, res_y[2], 'black', label = 'y3(x) Euler')
#subplot.plot(res_x, res_y[3], 'yellow', label = 'y4(x) Euler')
subplot.legend()
fig.show()
    
