import sys
import numpy as np
import matplotlib.pyplot as plt
 
from scipy import integrate





TOLE = 0.0005
gamma = 42.7
X_accuracy = 0.005
X_min = 2**(1/6)

N_integral = 150 


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

def calc_k2(x, gamma, epsilon):
    return (gamma**2) * (epsilon - LenardJones(x))


def calc_y_np1(y_nm1, y_n, dx, k2nm1, k2n, k2np1):
    const1 = 1 / (1 + (dx**2) * k2np1 / 12)
    const2 = 1 - (5 * (k2n) * (dx**2) / 12)
    const3 = 1 + ((dx**2) * (k2nm1) / 12)

    ynp1 = const1 * (2 * const2 * y_n - const3 * y_nm1)
    return ynp1

def Numerov_calc(y_init, x, dx, gamma, epsilon):
    y_res = [y_init[0], y_init[1]]
    
    y_nm1 = y_init[0] # y index n-1
    y_n = y_init[1] # y index n+1
    
    np1 = 2 # индекс n+1, 0 и 1 для x пропускаем
    len_x = len(x)
    while (np1 <= len_x - 1):
        k2_np1 = (calc_k2(x[np1], gamma, epsilon)) # k^2 индекс n+1
        print(k2_np1)
        k2_n = (calc_k2(x[np1-1], gamma, epsilon))
        print(k2_n)
        k2_nm1 = (calc_k2(x[np1-2], gamma, epsilon))
        print(k2_nm1)
        
        y_np1 = calc_y_np1(y_nm1, y_n, dx, k2_nm1, k2_n, k2_np1)
        y_res.append(y_np1)
        np1 +=1
    
    return y_res  

V0 = 1
a = 1 # Единица длины
m = 1
h_ = 1
gamma = (2*m*(a**2)*V0) / (h_**2)
X_MIN = 0.1 # Левая граница класс. запрещ. области psi(x_min) = 0
X_MAX = 3 # Правая граница класс. запрещ. области psi(x_max) = 0


# Расчёт S при малой E для нахождения n состояний
E = -TOLE
epsilon = E / V0

[XIN, XOUT] = calc_XIN_XOUT(E)

X_m = (XIN + XOUT) / 2 # Точка сшивки

N_pts_integral = 100
X_FROM = X_MIN
X_TO = X_m
x = np.linspace(X_FROM, X_TO, N_pts_integral)

dx = (X_FROM - X_TO) / (N_pts_integral - 1)

y_0 = 0 # y(x_MIN)
y_1 = 1 # Произвольная константа
y_init_cond = [y_0, y_1]
y = Numerov_calc(y_init_cond, x, dx, gamma, epsilon)
fig =  plt.figure()
subplot = fig.add_subplot()
subplot.plot(x, y)
fig.show()


    
    





    
    

