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

#def func(E)

def calc_s(XIN, XOUT, E):
    '''h = (XOUT-XIN)/N_integral
    
    func = lambda x: (abs(E-LenardJones(x)))**(1/2)
    
    integral = integrate.quad(func, XIN, XOUT)
    
    return integral[0]*gamma'''
    # функция вида (sqrt(E-LenardJones(x)))
    h = (XOUT-XIN)/N_integral
    x = np.linspace(XIN, XOUT, int((XOUT-XIN)/h))
    func = lambda x: (abs(E-LenardJones(x)))**(1/2)
    y = func(x)
    
    i = 0
    integral=0
    while i<= len(x)-2:
        y_0 = y[i+1]
        y_prev = y[i]
        y_next = y[i+1]
        J1 = (h/3)*(y_next+y_prev+4*y_0)
        integral+=J1
        i+=3
        
    return integral*gamma
    
       

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


def calc_Energy(N, E1, E2, DE, F1, s, XIN, XOUT):
    
    #N - energy level
    while abs(DE)>= TOLE:
        E = E2
       # print('E=', E)
        [XIN_, XOUT_] = calc_XIN_XOUT(E)
       # print('xin xout', XIN, XOUT)
        s = calc_s(XIN_, XOUT_, E)
       # print('s=', s)
        F2 = s-(N+1/2)*np.pi
        if F2 == F1:
            print('F2==F1 2100 line')
        DE=-F2*(E2-E1)/(F2-F1)
        #print(DE)
        E1 = E2
        F1 = F2
        E2 = E1 + DE
        if E2 > 0:
            E2 = -TOLE
    #print('return')
    return [E, E2, F2, XIN_, XOUT_]
        
    
    
# Расчёт S при малой E для нахождения n состояний
E = -TOLE
[XIN, XOUT] = calc_XIN_XOUT(E)
print(XIN, XOUT)
s = calc_s(XIN, XOUT, E)
NMAX = s/np.pi - 0.5
NLEV = NMAX+1

results = [[], [], []]
xin = []
xout = []
energies = []

E1 = -1
F1 = -np.pi/2 #Функция f = s-(n+0.5)pi, см тетрадь
for N in range (0, int(NMAX)):
    E2 = E1 + abs(E1)/4
    DE = 2*TOLE
    [E, E2, F2, XIN, XOUT] = calc_Energy(N, E1, E2, DE, F1, s, XIN, XOUT)
    xin.append(XIN)
    xout.append(XOUT)
    energies.append(E)

def get_const_func(from_, to_, val):
    y = []
    x = np.linspace(from_, to_, 50)
    for i in range(0, len(x)):
        y.append(val)
    return [x, y]

fig = plt.figure()
subplot = fig.add_subplot(111)

for i in range(0, len(energies)):
    [x, y] = get_const_func(xin[i], xout[i], energies[i])
    subplot.plot(x, y, 'k', label='Energy level '+str(i))
    
subplot.legend()
fig.show()
    
    

