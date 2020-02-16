import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import Euler
import RungeKutt






def func1(x, y):
    return -2*y[0]+4*y[1]

def func2(x, y):
    return -y[0]+3*y[1]

def func3(x, y):
    return -2*y[0]+4*y[1]

def func4(x, y):
    return -y[0]+3*y[1]

####
#For planets simulation
def func_u(x, y): #y - массив значений функций в правой части системы
    return (-y[1]/(np.sqrt(y[1]*y[1] + y[3]*y[3])))

def func_x(x,y):
    return y[0]

def func_v(x,y):
    return (-y[3]/(np.sqrt(y[1]*y[1]+y[3]*y[3])))

def func_y(x,y):
    return y[2]




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
[res_x, res_y] = Euler.Euler(y_start, x_from, x_to, dx, list(func_arr))
[res_x_RK, res_y_RK] = RungeKutt.RungeKutt(y_start, x_from, x_to, dx, func_arr)

    
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



###################################
#simulations of planets

x_0 = 100
y_0 = 0
x_der_0 = 0 #derivative when t = 0
y_der_0 = 5

#Порядок уравнений - u', x', v', y' , где u = x', v = y'
start_cond = [x_der_0, x_0, y_der_0, y_0]

t_from = 0
dt = 0.2
N = 220 #steps numb
t_to = dt*N
planet_mass=1000


func_arr = [func_u, func_x, func_v, func_y]

[res_x, res_y] = Euler.Euler(start_cond, t_from, t_to, dt, list(func_arr))
[res_x_RK, res_y_RK] = RungeKutt.RungeKutt(start_cond, t_from, t_to, dt, list(func_arr))

    
fig = plt.figure()
subplot = fig.add_subplot(111)
subplot.plot(res_y[1], res_y[3], 'b--', label='Euler Planets')
subplot.plot(res_y_RK[1], res_y_RK[3], 'r', label='RungeKutt Planets')
subplot.legend()
fig.show()


