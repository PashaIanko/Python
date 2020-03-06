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
        print('eps=' + str(eps) + 'deps=' + str(deps))
        P1 = P2
    return eps

'''eps = epsilon(-0.01, 0.01, N)
print(eps)
x = np.linspace(0, 1, len(Psi))

fig = plt.figure()
subplot = fig.add_subplot(111)
subplot.plot(x, Psi)
fig.show()'''



def my_wavefunction(ep, k2, N, l2):
    Psi = [0, 1e-4] #initial cond
    #Psi[0] = 0
    #Psi[1] = 1e-4

    for i in range(2, N):
        #print(i)
        val = (2*(1-(5.0/12)* l2 * k2[i-1])*Psi[i-1]
                  -(1+(1.0/12) * l2 * k2[i-2]) * Psi[i-2]) / (1 + (1.0/12) * l2 * k2[i])
        Psi.append(val)
    return Psi


def my_epsilon(eps, deps, N, l2, x):
    #k2 = g2*(eps-v)
    k2 = calc_k2(eps, x)#g2*(eps - v)
    Psi = my_wavefunction(eps, k2, N, l2)
    P1 = Psi[N-1]
    eps += deps

    while(abs(deps) > 1e-12):
        k2 = calc_k2(eps, x)#g2*(eps - v)
        Psi = my_wavefunction(eps, k2, N, l2)
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



# Експоненциально нарастающее решение слева
eps_0 = -0.6
[XIN, XOUT] = calc_XIN_XOUT(eps_0)
X_m = (XIN + XOUT) / 2

N = 100
#Psi = np.zeros(N)
g2 = 200 # gamma^2
x = np.linspace(0.01, X_m, N)
k2 = calc_k2(eps_0, x)
print('k2=', k2)

dx2 = ((1.0) / (N-1))**2 # dx^2
Psi_left = my_wavefunction(eps_0, k2, N, dx2)
Psi_left = [-p for p in Psi_left]
fig = plt.figure()
subplot = fig.add_subplot(111)
subplot.plot(x, Psi_left)
fig.show()


# Експоненциально нарастающее решение справа (Интегрирование вовнутрь класс. разреш. области)
x = np.linspace(3, X_m, N)
k2 = calc_k2(eps_0, x)

print('k2 right=', k2)

Psi_right = my_wavefunction(eps_0, k2, N, dx2)

fig = plt.figure()
subplot = fig.add_subplot(111)
subplot.plot(x, Psi_right)
fig.show()




        
    
    
    

