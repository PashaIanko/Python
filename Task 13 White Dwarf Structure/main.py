import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import Euler
import RungeKutt


def gamma(rho_):
    return (rho_)**(2) / (3 * (1 + rho_**2))**(1 / 2)

def func_m_(r_, y): # y - массив из текущих значений тех величин, что в левой части СДУ (под дифференциалом)
    return ((r_**2) * abs(y[1]))

def func_rho_(r_, y):
    return -(y[0] * abs(y[1])) / (gamma((y[1])**(1/3)) * r_**2)


def calc_density_distr(Ye, rho_c):
    # White Dwarf Simulation
    #Ye = 100
    #rho_c = 1000000 #15000 #[г / см^3] - плотность в центре Солнца

    #rho_c [г/см^3]
    rho_0 = 9.79*(10**5) * (Ye**(-1)) #[г * см-3] #1

    _r_from = 0.01 # r_ - r с чертой (см. методичку)
    d_r = 0.01
    _r_to = 1
    N = (_r_to - _r_from)/d_r #steps numb
    

    # Initial Conditions
    _m_0 = 0.01
    _rho_0 = rho_c/rho_0

    start_cond = [_m_0, _rho_0]


    func_arr = [func_m_, func_rho_]


    [res_r_RK, res_y_RK] = RungeKutt.RungeKutt(start_cond, _r_from, _r_to, d_r, list(func_arr))

    res_m = res_y_RK [0]
    res_rho = res_y_RK[1]

    return [res_m, res_rho, res_r_RK]

    fig = plt.figure()
    fig.suptitle('Ye = '+str(Ye)+' rho_c= '+str(rho_c))
    subplot = fig.add_subplot(121)
    subplot.plot(res_r_RK, res_m, label='mass, m/M0')
    subplot.legend()

    subplot = fig.add_subplot(122)
    subplot.plot(res_r_RK, res_rho, 'r--', label='Density, VS r/R0')
    subplot.legend()
    fig.show()

def plot_density(subplot_density, subplot_mass, Ye, rho_c):
    [res_m, res_rho, res_r] = calc_density_distr(Ye, rho_c)
    res_rho = [abs(x) for x in res_rho]
    res_m = [abs(y) for y in res_m]# abs(res_m)
    subplot_density.plot(res_r, res_rho, label = 'Density, rho_c='+str(rho_c)+', Ye='+str(Ye))
    subplot_density.legend()

    subplot_mass.plot(res_r, res_m, label = 'Density, rho_c='+str(rho_c)+', Ye='+str(Ye))
    subplot_mass.legend()

fig = plt.figure()
subplot_density = fig.add_subplot(121)
subplot_density.set_title('density')

subplot_mass = fig.add_subplot(122)
subplot_mass.set_title('mass')

plot_density(subplot_density, subplot_mass, 100, 1)
plot_density(subplot_density, subplot_mass, 100, 100)
plot_density(subplot_density, subplot_mass, 100, 1000)
plot_density(subplot_density, subplot_mass, 100, 10000)
plot_density(subplot_density, subplot_mass, 100, 100000)
plot_density(subplot_density, subplot_mass, 100, 1000000)
plot_density(subplot_density, subplot_mass, 1000, 1000000)


fig.show()


'''fig = plt.figure()
subplot = fig.add_subplot(111)
subplot.plot(res_y_RK[1], res_y_RK[3], 'r', label='RungeKutt Planets')
subplot.legend()
fig.show()'''









