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

def func_test_0(x, y):
    return 2*(x**2+y[0])

def func_test_1(x, y):
    return 2*(x**2+y[1])

def func_test_2(x, y):
    return 2*(x**2+y[2])

def func_test_3(x, y):
    return 2*(x**2+y[3])

####
#For planets simulation
def func_u(x, y): #y - массив значений функций в правой части системы
    return -10000*y[1]/((y[1]*y[1] + y[3]*y[3])**(3/2))

def func_x(x,y):
    return y[0]

def func_v(x,y):
    return -y[3]*10000/((y[1]*y[1] + y[3]*y[3])**(3/2))

def func_y(x,y):
    return y[2]


#vector function
'''func_arr = [func1, func2, func3, func4]

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
fig.show()'''


def calc_res(x):
    res = []
    for i in range (0, len(x)):
        val = 1.5*np.exp(2*x[i])-(x[i]*x[i]) - x[i] - 0.5
        res.append(val)
    return res

def fun(t, y):

    return 2*(t*t+y);


def seminar_alg(dt, from_, to_):
    
    t=from_
    y1=1
    y2=1
    y3=1
    k1=0
    k2=0
    k3=0
    k4=0
    
    t=from_;
    y1=1;
    y2=1;
    y3=1;
    
    y_Euler = []
    y_RK = []
    y_theory = []
    time = []
    
    while t+dt<to_:
    
#метод Эйлера
        y1=y1+dt*fun(t,y1)
#метод Рунге-Кутта 4-го порядка
        k1=dt*fun(t,y2)
        k2=dt*fun(t+dt/2,y2+k1/2)
        k3=dt*fun(t+dt/2,y2+k2/2)
        k4=dt*fun(t+dt,y2+k3)
        y2=y2+(k1+2*(k2+k3)+k4)/6
        t+=dt
#аналитическое решение
        y3=1.5*np.exp(2*t)-t*t-t-0.5
        
        y_Euler.append(y1)
        y_RK.append(y2)
        y_theory.append(y3)
        #print("y2, y3=", y2, y3)
        time.append(t)
    return [y_Euler, y_RK, y_theory, time]

###########################
# Тест из методички
from_ = 0
to_ = 1
dt = 0.1
func_arr = [func_test_0, func_test_1, func_test_2, func_test_3]
y_start = [1, 1, 1, 1]
[res_x, res_y] = Euler.Euler(y_start, from_, to_, dt, func_arr)
[res_x_RK, res_y_RK] = RungeKutt.RungeKutt(y_start, 0, 1, 0.1, func_arr)

[y_Euler, y_RK, y_theory, time] = seminar_alg(dt, from_, to_)

print('Seminar Euler\n', y_Euler, '\n')
print('My Euler\n', res_y, '\n')

print('Seminar RK\n', y_RK, '\n')
print('My RK\n', res_y_RK, '\n')


'''fig = plt.figure()
subplot = fig.add_subplot(121)
subplot.set_title('методичка')

subplot.plot(time, y_theory, 'black', label='Seminar theory')
subplot.plot(time, y_Euler, 'm', label = 'Seminar Euler')
subplot.plot(res_x, res_y[0], 'g--', label = 'y1(x) My Euler')
subplot.legend()

subplot = fig.add_subplot(122)
subplot.plot(time, y_theory, 'black', label='Seminar theory')
subplot.plot(time, y_RK, 'm', label = 'Seminar Runge Kutt')
subplot.plot(res_x_RK, res_y_RK[0], 'g--', label = 'y3(x) My RungeKutt')
subplot.legend()

fig.show()'''



###################################
#simulations of planets

'''x_0 = 100
y_0 = 0
x_der_0 = 0 #derivative when t = 0
y_der_0 = 5

#Порядок уравнений - u', x', v', y' , где u = x', v = y'
start_cond = [x_der_0, x_0, y_der_0, y_0]

t_from = 0
dt = 0.2
N = 240 #steps numb
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
fig.show()'''


