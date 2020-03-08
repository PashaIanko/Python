import sys
import numpy as np

import matplotlib.pyplot as plt
 
from scipy import integrate


    
def LenardJones(x):
    return 4*(x**(-12) - x**(-6))

X_accuracy = 0.005
X_min = 2**(1/6)

def calc_XIN_XOUT(E):
    dx = 0.1
    XIN = X_min # Поиск влево, точка поворота
    while dx>X_accuracy:
                
        if LenardJones(XIN)>=E:
            XIN+=dx
            dx = dx/2
        XIN= XIN-dx
      
    XOUT = X_min
    dx=0.1
    while dx>X_accuracy:
        
        if(LenardJones(XOUT)>=E):
            XOUT-=dx
            dx=dx/2
    
        XOUT+=dx
    
    return [XIN, XOUT]



N = 1000
Psi = np.zeros(N)
g2 = 200
v = (-1)*np.ones(N)
ep = -0.9
k2 = g2*(ep-v)
l2 = ((1.0) / (N-1))**2

def wavefunction(ep, k2, N):
    Psi[0] = 0
    Psi[1] = 1e-4

    for i in range(2, N):
        Psi[i] = (2*(1-(5.0/12)* l2 * k2[i-1])*Psi[i-1]
                  -(1+(1.0/12) * l2 * k2[i-2]) * Psi[i-2]) / (1 + (1.0/12) * l2 * k2[i])
    return Psi

'''Psi = wavefunction(ep, k2, N)
x = np.linspace(0, 1, len(Psi))

fig = plt.figure()
subplot = fig.add_subplot(111)
subplot.plot(x, Psi)
fig.show()'''

def epsilon(eps, deps, N):
    k2 = g2*(eps-v)
    wavefunction(Psi, k2, N)
    P1 = Psi[N-1]
    eps += deps

    while(abs(deps) > 1e-12):
        k2 = g2*(eps - v)
        wavefunction(Psi, k2, N)
        P2 = Psi[N-1]

        if P1*P2 <0:
            deps = -deps/2
        eps = eps + deps
        #print('eps=' + str(eps) + 'deps=' + str(deps))
        P1 = P2
    return eps

'''eps = epsilon(-0.01, 0.01, N)
print(eps)
x = np.linspace(0, 1, len(Psi))

fig = plt.figure()
subplot = fig.add_subplot(111)
subplot.plot(x, Psi)
fig.show()'''



def my_wavefunction(ep, k2, N, l2, init_cond):
    #Psi = [0, 1e-4] #initial cond
    Psi = [init_cond[0], init_cond[1]]
    #Psi[0] = 0
    #Psi[1] = 1e-4

    for i in range(2, N):
        #print(i)
        val = (2*(1-(5.0/12)* l2 * k2[i-1])*Psi[i-1]
                  -(1+(1.0/12) * l2 * k2[i-2]) * Psi[i-2]) / (1 + (1.0/12) * l2 * k2[i])
        Psi.append(val)
    return Psi


def my_epsilon(eps, deps, N, l2, x, init_cond):
    #k2 = g2*(eps-v)
    k2 = calc_k2(eps, x)#g2*(eps - v)
    Psi = my_wavefunction(eps, k2, N, l2, init_cond)
    P1 = Psi[N-1]
    eps += deps

    while(abs(deps) > 1e-12):
        k2 = calc_k2(eps, x)#g2*(eps - v)
        Psi = my_wavefunction(eps, k2, N, l2, init_cond)
        P2 = Psi[N-1]

        if P1*P2 <0:
            print('shit')
            deps = -deps/2
        eps = eps + deps
        #print('eps=' + str(eps) + 'deps=' + str(deps))
        P1 = P2
    return eps

def calc_k2(eps, x):
    k2_res = []
    for i in range(0, len(x)):
        k2_ = g2*(eps - LenardJones(x[i]))
        #print('k2=' , k2_)
        k2_res.append(k2_)
    return k2_res

# Lenard Jones potential
'''eps_0 = -0.6
[XIN, XOUT] = calc_XIN_XOUT(eps_0)
X_m = (XIN + XOUT) / 2

N = 100
#Psi = np.zeros(N)
g2 = 200 # gamma^2
x = np.linspace(XIN, XOUT, N)
k2 = calc_k2(eps_0, x)

dx2 = ((1.0) / (N-1))**2 # dx^2
Psi = my_wavefunction(eps_0, k2, N, dx2)

fig = plt.figure()
subplot = fig.add_subplot(111)
subplot.plot(x, Psi)
fig.show()

eps = my_epsilon(eps_0, 0.01, N, dx2, x)

fig = plt.figure()
subplot = fig.add_subplot(111)
subplot.plot(x, Psi)
fig.show()'''

def calc_Psi(eps_0, N, g2, FROM, TO, XIN, XOUT, init_cond):
    
    
    X_m = (XIN + XOUT) / 2

    #TO = X_m     
    x = np.linspace(FROM, TO, N)
    k2 = calc_k2(eps_0, x)
    
   
    dx = (FROM - TO) / (N-1)
    dx2 =  dx ** 2
    Psi = my_wavefunction(eps_0, k2, N, dx2, init_cond)
    return [Psi, x, X_m]
    
   
# Експоненциально нарастающее решение слева
eps_0 = -0.6
[XIN, XOUT] = calc_XIN_XOUT(eps_0)

X_m = (XIN + XOUT) / 2
[Psi_left, x_left, X_m] = calc_Psi(eps_0, 100, 200, 0.01, X_m, XIN, XOUT, [0, 1e-4])
x_continue = x_left[len(x_left) - 2] # значение в которой надо сосчитать
print('x_continue'+str(x_continue))
Psi_left = [-p for p in Psi_left]

[Psi_right, x_right, X_m] = calc_Psi(eps_0, 100, 200, 3, x_continue, XIN, XOUT, [0, 1e-4])

fig = plt.figure()
subplot = fig.add_subplot(121)
subplot.plot(x_left, Psi_left, label = 'Psi left')
subplot.legend()
subplot = fig.add_subplot(122)
subplot.plot(x_right, Psi_right, label = 'Psi right')
subplot.legend()
fig.show()


# Експоненциально нарастающее решение справа (Интегрирование вовнутрь класс. разреш. области)
# - колво точек - N, но строим до X_m - h



'''fig = plt.figure()
subplot = fig.add_subplot(121)
subplot.plot(x_right, Psi_right, label = 'Psi right')
subplot.legend()
#fig.show()'''


# нормировка, чтобы функции сшились в X_m
right_val = Psi_right[len(Psi_right)-1]
left_val = Psi_left[len(Psi_left)-1]
ratio = right_val / left_val
Psi_left = [p*ratio for p in Psi_left]
x = np.linspace(0.01, X_m, len(Psi_left))
subplot = fig.add_subplot(122)
subplot.plot(x, Psi_left, label = 'Psi left')
subplot.legend()
fig.show()

def my_epsilon_(eps, deps, N):

    [XIN, XOUT] = calc_XIN_XOUT(eps)

    X_m = (XIN + XOUT) / 2


    [Psi_left, x_left, X_m] = calc_Psi(eps, N, 200, 0.01, X_m, XIN, XOUT, [0, 1e-4])
    x_continue = x_left[len(x_left) - 2] # значение в которой надо сосчитать
       
    Psi_left = [-p for p in Psi_left]
    [Psi_right, x_right, X_m] = calc_Psi(eps, N, 200, 3, x_continue, XIN, XOUT, [0, 1e-4])

    # Нормировка
    right_val = Psi_right[len(Psi_right)-1]
    left_val = Psi_left[len(Psi_left)-1]
    ratio = right_val / left_val
    Psi_left = [p*ratio for p in Psi_left]


    val_to_check_right = Psi_right[len(Psi_right) - 1]
    val_to_check_left = Psi_left[len(Psi_left) - 2]

    diff1 = val_to_check_right - val_to_check_left
    print('diff1', diff1)
    
    while(abs(deps) > 1e-12):
        [XIN, XOUT] = calc_XIN_XOUT(eps)
        X_m = (XIN + XOUT) / 2
        
        [Psi_left, x_left, X_m] = calc_Psi(eps, N, 200, 0.01, X_m, XIN, XOUT, [0, 1e-4])
        x_continue = x_left[len(x_left) - 2] # значение в которой надо сосчитать
        
        Psi_left = [-p for p in Psi_left]

        [Psi_right, x_right, X_m] = calc_Psi(eps, N, 200, 3, x_continue, XIN, XOUT, [0, 1e-4])

        # Нормировка
        right_val = Psi_right[len(Psi_right)-1]
        left_val = Psi_left[len(Psi_left)-1]
        ratio = right_val / left_val
        Psi_left = [p*ratio for p in Psi_left]


        val_to_check_right = Psi_right[len(Psi_right) - 1]
        val_to_check_left = Psi_left[len(Psi_left) - 2]
       # print('vals to check:', val_to_check_right, val_to_check_left)

        diff2 = val_to_check_right - val_to_check_left


        if diff1*diff2 <0:
            #print('shit')
            deps = -deps/2
        eps = eps + deps
        #print('eps=' + str(eps) + 'deps=' + str(deps))
        diff1 = diff2
    return [eps, Psi_left, x_left, Psi_right, x_right]



def calc_real_energy(eps):
    [eps_real, Psi_left, x_left, Psi_right, x_right] = my_epsilon_(eps, 0.05, 100)
    [XIN, XOUT] = calc_XIN_XOUT(eps_real)
    return [Psi_left, x_left, Psi_right, x_right, eps_real, XIN, XOUT]

# точное значение энергии, через сшивку производных
def test(eps):
    [eps_real, Psi_left, x_left, Psi_right, x_right] = my_epsilon_(eps, 0.05, 100)
    print('eps=', eps, ' eps actual=\n', eps_real)
    fig = plt.figure()
    subplot = fig.add_subplot(121)
    subplot.plot(x_left, Psi_left, label = 'Psi left')
    
    subplot.plot(x_right, Psi_right, label = 'Psi right, eps='+str(eps))
    subplot.legend()
    fig.show()
    return [Psi_left, x_left, Psi_right, x_right, eps_real]
    


def find_index(x, value, accuracy):
    for i in range(0,len(x)):
        if abs(x[i] - value) <= accuracy:
            return i
    return -1

'''# Тест, считается ли волновая функция на промежутке корректно

eps = -0.9
[Psi_left, x_left, Psi_right, x_right, eps_real] = test(eps)
[XIN, XOUT] = calc_XIN_XOUT(eps_real)
y_init_index = find_index(Psi_left, XIN, 0.05)

y_init0 = Psi_left[y_init_index-1]
y_init1 = Psi_left[y_init_index]

[Psi, x, xm ] = calc_Psi(eps_real, 100, 200, XIN, XOUT, XIN, XOUT, [y_init0, y_init1])
       
    
fig = plt.figure()
subplot = fig.add_subplot(111)
subplot.plot(x, Psi, label = 'Psi')
subplot.legend()
fig.show()'''

def get_const_func(from_, to_, val):
    y = []
    x = np.linspace(from_, to_, 50)
    for i in range(0, len(x)):
        y.append(val)
    return [x, y]

def plot_energy_levels(subplot, eps):
    [Psi_left, x_left, Psi_right, x_right, eps_real, XIN, XOUT] = calc_real_energy(eps)
    #[eps_real, Psi_left, x_left, Psi_right, x_right, XIN, XOUT] = calc_real_energy(eps)
    print('eps real', eps_real)
    [x, y] = get_const_func(XIN, XOUT, eps_real)
    subplot.plot(x, y, label = 'Energy Level, eps = '+str(eps))

fig = plt.figure()
subplot = fig.add_subplot(121)
plot_energy_levels(subplot, -0.9)
plot_energy_levels(subplot, -0.85)
plot_energy_levels(subplot, -0.75)
plot_energy_levels(subplot, -0.8)
plot_energy_levels(subplot, -0.7)
plot_energy_levels(subplot, -0.6)
plot_energy_levels(subplot, -0.65)
plot_energy_levels(subplot, -0.5)
plot_energy_levels(subplot, -0.3)
plot_energy_levels(subplot, -0.2)

fig.show()

'''test(-0.9)
test'(-0.8)
test(-0.7)
test(-0.6)
test(-0.5)
test(-0.4)
test(-0.3)'''



    

