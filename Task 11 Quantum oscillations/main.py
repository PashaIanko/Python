import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def LenardJones(x):
    return 4*(x**(-12)-x**(-6))

def calc_S(E, gamma, Npts, x_acc, x_min):
    xin = x_min
    dx = 0.1
    while dx>x_acc:
        xin = xin-dx
        if LenardJones(xin)<E:
            break
        else:
            xin += dx
            dx=dx/2
    xout = x_min
    dx = 0.1
    while dx>x_acc:
        xout+=dx
        if LenardJones(xout) < E:
            break
        else:
            xout -=dx
            dx = dx/2
        
    H = (xout-xin)/Npts #Формула Симпсона
    Sum = 0
    Sum += np.sqrt(E-LenardJones(xin+H))

    FAC = 2
    for i in range(2, Npts-2):
        x = xin +i*H
        if FAC==2:
            FAC=4
        else:
            FAC=2
        Sum=Sum+FAC+np.sqrt(E-LenardJones(x))

    Sum=Sum+np.sqrt(E-LenardJones(xout-H))
    Sum=Sum*H/3

    Sum=Sum+np.sqrt(E-LenardJones(xin+H))*2*H/3
    Sum=Sum+np.sqrt(E-LenardJones(xout-H))*2*H/3
    S=gamma*Sum

    return [S, xin, xout]

def calc_levels(TOLE, MAX, gamma, Npts, TOLX, XMIN):
    E = -TOLE
    # Ниже - рассчёт S в единицах 2*(h/2pi)
    [S, xin, xout] = calc_S(E, gamma, Npts, TOLX, XMIN)

    NMAX = S/np.pi - 0.5
    NLEV = NMAX + 1 #NLEV - число уровней
    print("число уровней =", NLEV)
    if NLEV <= MAX:
        print("Число уровней должно быть меньше ", MAX)
        print("Гамма нужна меньше")
    return [NLEV, NMAX, S]

def calc_energy_level(N, DE, E1, E2, F1, n, S, XIN, XOUT, TOLE, gamma, Npts, TOLX, XMIN, En, x_in, x_out):
    E=0
    while abs(DE)>= TOLE:
        E = E2
        [S_, XIN_, XOUT_] = calc_S(E, gamma, Npts, TOLX, XMIN)
        F2 = S_-(N + 0.5)*np.pi
        if F2==F1:
            print("F2 == F1")
            #2100 строка
        else:
            DE = -F2*(E2-E1)/(F2-F1)
            E1=E2
            F1=F2
            E2=E1+DE
            if E2>0:
                E2=-TOLE
    En.append(E)
    print("XIN=", XIN, "XOUT=", XOUT)
    x_in.append(XIN_)
    x_out.append(XOUT_)
    return F2

def calc_K(En, XIN, XOUT, gamma):
    K = []
    for i in range (0, len(En)):
        x = XIN[i]
        k = gamma*np.sqrt(En[i]-LenardJones(x))
        K.append(k)
    return K

def draw(K, XIN):
    for i in range(0, len(K)):
        if not (np.isnan(K[i])):
            plt.scatter(XIN[i], K[i], color='r')
    plt.show()



        
MAX = 100 #максимальное число уровней для расчёта
En = []
XIN = []
XOUT = []

TOLX = 0.005 #Погрешность точек поворота
TOLE = 0.005

Npts = 40 #число узлов интегррования
XMIN = 2**(1/6) #Положение минимума потенциала
gamma = 21.7

#По заданному значению гамма нужно вычислить NMAX и NLEV уровней
# NMAX - количество всех уровней

#ввод гамма

###

# На этом этапе мы получили NLEV
[NLEV, NMAX, S] = calc_levels(TOLE, MAX, gamma, Npts, TOLX, XMIN)

E1 = -1
F1 = 0-np.pi/2

for n in range (0, int(NMAX)):
    E2=E1+abs(E1)/4
    DE=2*TOLE
    F2 = calc_energy_level(0, DE, E1, E2, F1, n, S, XIN, XOUT, TOLE, gamma, Npts, TOLX, XMIN, En, XIN, XOUT)
    E1=E2
    F1 = F2-np.pi
    
K=calc_K(En, XIN, XOUT, gamma)

draw(K, XIN)

