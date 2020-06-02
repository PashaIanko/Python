import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def compute_heat_states(τ, τ_total):
    return int(np.ceil(τ_total / τ) + 1.0)


def average(T_cell, i, j, cell_height, cell_width):
    # look for neighbours
    res = 0
    counter = 0
    if (i-1 >= 0):
        if(j-1 >= 0):
            res += T_cell[i-1][j-1] + T_cell[i-1][j] + T_cell[i][j-1]
            
            counter+=3
        if(j+1 < cell_height):
            res += T_cell[i-1][j+1]
            counter +=1
    if(i+1 < cell_width):
        if(j-1 >=0):
            res += T_cell[i+1][j-1] + T_cell[i+1][j]
            counter +=2
    if(j+1 < cell_height):
        res += T_cell[i][j+1]
        counter +=1
        if(i+1 < cell_width):
            res += T_cell[i+1][j+1]
            counter+=1
    res = res / counter
    return res
            
        
        

def calc_average(T_cell, cell_height, cell_width):
    T_cell_res = np.zeros([cell_height, cell_width], dtype = To.dtype)
    for i in range(0, cell_height):

            for j in range(0, cell_width):
                #print('before average()')
                res_average = average(T_cell, i, j, cell_height, cell_width)
                #print('T cell res, i=', i, 'j=', j)
                T_cell_res[i, j] = res_average
    return T_cell_res

def compute_heat_process_progonka(To, N, k):
    h = np.shape(To)[0]
    w = np.shape(To)[1]


    # Результирующий массив мешей - распределение температуры с течением времени
    #print('cell height', h)
    #print('cell width', w)
    #print('N=', N)
    T = np.zeros([N, h, w], dtype=To.dtype)
    T[0, 1:-1, 1:-1] = To[1:-1, 1:-1]
    #print('T initial = ', T[0])
    tmp = np.zeros([h, w], dtype=To.dtype)

  
    for m in (range(1, N)):
        #print(T[m-1])
        T[m] = calc_average(T[m-1], h, w)
        
                
                
    return T

To = np.zeros([11, 11], dtype=np.float32)
To[5][5] = 5000000.
To[5][6] = 5000000.
To[5][4] = 5000000.
To[4][5] = 5000000.
To[6][5] = 5000000.


h = 0.01
simulation_time = 100.0

ρ_fe = 7800 # Kg/m^3
c_fe = 150 # для железа - 450 J/(Kg*°K)
λ_fe = 200#80 - для железа W/(m*°K)

ρ = ρ_fe
c = c_fe
λ = λ_fe

τ = 0.01 # s
τ_total = 60.0 # s


σ = λ / (ρ * c) # m^2/s
k = τ*σ / (h ** 2)
i = 0

def updatefig(*args):
    global i
    if i < T.shape[0] - 1:
        print(i)
        i += 1
        im.set_array(T[i])
        #print('im=', type(im), im)
        #print('T[i]=' , T[i])
    return im,


#
N = compute_heat_states(τ, τ_total)
T = compute_heat_process_progonka(To, N, k)
#print('type T = ', type(T), 'T=', T)
#
fig2 = plt.figure()
#
i = 0
im = plt.imshow(T[0], cmap='hot', animated=True)
#
ani = animation.FuncAnimation(fig2, updatefig, interval=3000, blit=True)
plt.show()
