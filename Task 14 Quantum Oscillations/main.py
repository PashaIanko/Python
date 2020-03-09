import sys
import numpy as np

import matplotlib.pyplot as plt
 
from scipy import integrate



X_accuracy = 0.005
X_min = 2**(1/6)

    
def LenardJones(x):
    return 4*(x**(-12) - x**(-6))

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


def my_wavefunction(ep, k2, N, l2, init_cond):
    
    Psi = [init_cond[0], init_cond[1]]
    
    for i in range(2, N):
        val = (2*(1-(5.0/12)* l2 * k2[i-1])*Psi[i-1]
                  -(1+(1.0/12) * l2 * k2[i-2]) * Psi[i-2]) / (1 + (1.0/12) * l2 * k2[i])
        Psi.append(val)
    return Psi


def calc_k2(eps, x, FROM, TO):
    k2_res = []
    for i in range(0, len(x)):
        if (FROM <= x[i] <= TO or TO <= x[i] <= FROM):
            k2_ = g2*(eps - LenardJones(x[i]))
            k2_res.append(k2_)
    return k2_res


def calc_Psi(eps_0, N, g2, x, FROM, TO, init_cond):
    
    #X_m = (XIN + XOUT) / 2

    #TO = X_m     
    #x = np.linspace(FROM, TO, N)
    x_res = []
    for i in range(0, len(x)):
        if (FROM <= x[i] <= TO or TO <= x[i] <= FROM):
            x_res.append(x[i])
    
    k2 = calc_k2(eps_0, x_res, FROM, TO)
    
    #print('x_res=\n' , x_res)
    dx = abs(x_res[1]-x_res[0])#(FROM - TO) / (N-1)
    dx2 =  dx ** 2
    ''' print(k2)'''
    
    Psi = my_wavefunction(eps_0, k2, len(x_res), dx2, init_cond)
    return [Psi, x_res]#[Psi, x, X_m]
    
   
# Експоненциально нарастающее решение слева
eps_0 = -0.2
from_ = 0.01
to_ = 3
N = 200
[XIN, XOUT] = calc_XIN_XOUT(eps_0)
X_m = (XIN + XOUT) / 2


#print('x_left' , x_left)


x = np.linspace(from_, X_m, N)
gamma2 = 200
x_continue = x[len(x) - 2]
print('x_cont=' , x_continue)
[Psi_left, x_left] = calc_Psi(eps_0, N, gamma2, x, from_, X_m, [0, 1e-4])


x = np.linspace(to_, X_m, N)
x = np.append(x, x_continue)
[Psi_right, x_right] = calc_Psi(eps_0, N, gamma2, x, to_, x_continue, [0, 1e-4])


fig = plt.figure()
subplot = fig.add_subplot(121)
subplot.plot(x_left, Psi_left, label = 'Psi left calibrated')
subplot.legend()
subplot = fig.add_subplot(122)
subplot.plot(x_right, Psi_right, label = 'Psi right calibrated')
subplot.legend()
fig.show()


# нормировка, чтобы функции сшились в X_m
right_val = Psi_right[len(Psi_right) - 2]
print('right_val=', right_val)

left_val = Psi_left[len(Psi_left)-1]
print('left_val=', left_val)
ratio = (right_val / left_val)
Psi_left = [p*ratio for p in Psi_left]
x = np.linspace(0.01, X_m, len(Psi_left))
fig = plt.figure()
subplot = fig.add_subplot(121)
subplot.plot(x, Psi_left, label = 'Psi left')
subplot.legend()

subplot = fig.add_subplot(122)
x = np.linspace(X_m, 3, len(Psi_right))
subplot.plot(x, Psi_right, label = 'Psi right')
subplot.legend()
fig.show()

def integral(x, y):
    i = 0
    integral=0
    h = abs(x[1] - x[0])
    while i<= len(x)-2:
        y_0 = y[i+1]
        y_prev = y[i]
        y_next = y[i+1]
        J1 = (h/3)*(y_next+y_prev+4*y_0)
        integral+=J1
        i+=3
    return integral

def Simpson_integrate(x_left, Psi_left, x_right, Psi_right):
    Psi_left = [p*p for p in Psi_left]
    Psi_right = [p*p for p in Psi_right]
    half_left = integral(x_left, Psi_left)
    half_right = integral(x_right, Psi_right)
    '''i = 0
    integral=0
    h = abs(x_left[1] - x_left[0])
    while i<= len(x)-2:
        y_0 = Psi_left[i+1]
        y_prev = Psi_left[i]
        y_next = Psi_left[i+1]
        J1 = (h/3)*(y_next+y_prev+4*y_0)
        integral+=J1
        i+=3'''
        
    return half_left + half_right

def my_epsilon_(eps, deps, N, g2, FROM, TO):

    [XIN, XOUT] = calc_XIN_XOUT(eps)

    X_m = (XIN + XOUT) / 2

    x = np.linspace(FROM, X_m, N)

    x_continue = x[len(x) - 2]
    print('x_cont=' , x_continue)
    [Psi_left, x_left] = calc_Psi(eps, N, gamma2, x, FROM, X_m, [0, 1e-4])
       
    x = np.linspace(TO, X_m, N)
    x = np.append(x, x_continue)
    [Psi_right, x_right] = calc_Psi(eps, N, gamma2, x, TO, x_continue, [0, 1e-4])

    
    # Нормировка
    right_val = Psi_right[len(Psi_right)-2]
    left_val = Psi_left[len(Psi_left)-1]
    ratio = (right_val / left_val)
    Psi_left = [p*ratio for p in Psi_left]


    val_to_check_right = Psi_right[len(Psi_right) - 1]
    val_to_check_left = Psi_left[len(Psi_left) - 2]

    diff1 = val_to_check_right - val_to_check_left
    #print('diff1', diff1)
    
    while(abs(deps) > 1e-12):
        x = np.linspace(FROM, X_m, N)

        x_continue = x[len(x) - 2]
        print('x_cont=' , x_continue)
        [Psi_left, x_left] = calc_Psi(eps, N, gamma2, x, FROM, X_m, [0, 1e-4])
       
        x = np.linspace(TO, X_m, N)
        x = np.append(x, x_continue)
        [Psi_right, x_right] = calc_Psi(eps, N, gamma2, x, TO, x_continue, [0, 1e-4])

    
        # Нормировка
        right_val = Psi_right[len(Psi_right)-2]
        left_val = Psi_left[len(Psi_left)-1]
        ratio = (right_val / left_val)
        Psi_left = [p*ratio for p in Psi_left]


        val_to_check_right = Psi_right[len(Psi_right) - 1]
        val_to_check_left = Psi_left[len(Psi_left) - 2]
       # print('vals to check:', val_to_check_right, val_to_check_left)

        diff2 = val_to_check_right - val_to_check_left


        if diff1*diff2 <0:
            print('damn')
            deps = -deps/2
        eps = eps + deps
        #print('eps=' + str(eps) + 'deps=' + str(deps))
        diff1 = diff2

    # нормировка волновых функций на единицу
    Integral = Simpson_integrate(x_left, Psi_left, x_right, Psi_right)
    print('Integral = ', Integral)
    #Psi_left = [p / Integral for p in Psi_left]
    #Psi_right = [p / Integral for p in Psi_right]
    max_ = max(max(Psi_left), max(Psi_right))
    Psi_left = [p / max_ for p in Psi_left]
    Psi_right = [p / max_ for p in Psi_right]
    return [eps, Psi_left, x_left, Psi_right, x_right]



def calc_real_energy(eps, N, gamma2, FROM, TO):
    [eps_real, Psi_left, x_left, Psi_right, x_right] = my_epsilon_(eps, 0.05, N, gamma2, FROM, TO)
    [XIN, XOUT] = calc_XIN_XOUT(eps)
    return [Psi_left, x_left, Psi_right, x_right, eps_real, XIN, XOUT]


def get_const_func(from_, to_, val):
    y = []
    x = np.linspace(from_, to_, 50)
    for i in range(0, len(x)):
        y.append(val)
    return [x, y]

def plot_energy_levels(subplot_energies, subplot_wavefuncs, eps, N, gamma2, FROM, TO):
    [Psi_left, x_left, Psi_right, x_right, eps_real, XIN, XOUT] = calc_real_energy(eps, N, gamma2, FROM, TO)
    
    [x, y] = get_const_func(XIN, XOUT, eps)
    subplot_energies.plot(x, y, label = 'Energy Level, eps = '+str(eps))
    subplot_energies.legend()
    
    subplot_wavefuncs.plot(x_left, Psi_left, label = 'eps = '+str(eps))
    subplot_wavefuncs.plot(x_right, Psi_right, label = 'eps = '+str(eps))
    subplot_wavefuncs.legend()
    

fig = plt.figure()
subplot_energies = fig.add_subplot(121)
subplot_wavefuncs = fig.add_subplot(122)
plot_energy_levels(subplot_energies, subplot_wavefuncs, -0.9, 200, 200, 0.01, 3)
plot_energy_levels(subplot_energies, subplot_wavefuncs, -0.8, 200, 200, 0.01, 3)
plot_energy_levels(subplot_energies, subplot_wavefuncs, -0.7, 200, 200, 0.01, 3)
plot_energy_levels(subplot_energies, subplot_wavefuncs, -0.6, 200, 200, 0.01, 3)
plot_energy_levels(subplot_energies, subplot_wavefuncs, -0.5, 200, 200, 0.01, 3)
plot_energy_levels(subplot_energies, subplot_wavefuncs, -0.3, 200, 200, 0.01, 3)
plot_energy_levels(subplot_energies, subplot_wavefuncs, -0.2, 200, 200, 0.01, 3)

fig.show()





    

