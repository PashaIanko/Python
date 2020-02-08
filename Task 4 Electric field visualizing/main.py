import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
    
        

def E(q, r0, x, y):
    r_cube = np.hypot(x-r0[0], y-r0[1])**3
    return q * (x - r0[0]) / r_cube, q * (y - r0[1]) / r_cube


def orthogonal_vect(Vx, Vy):
    return -Vy, Vx

def PHI(q, r0, x, y):
    r = np.hypot(x-r0[0], y-r0[1])
    return q / r

# Grid of x, y points
nx, ny = 64, 64
x = np.linspace(-2, 2, nx)
y = np.linspace(-2, 2, ny)
X, Y = np.meshgrid(x, y)

nq = 2
charges = []
charge_left_val = -1
charge_right_val = 1
charges.append((charge_left_val, (1, 0)))
charges.append((charge_right_val, (-1, 0)))

#initializing
Ex, Ey = np.zeros((ny, nx)), np.zeros((ny, nx))
phi_x, phi_y = np.zeros((ny, nx)), np.zeros((ny, nx))
Ex_ort, Ey_ort = np.zeros((ny, nx)), np.zeros((ny, nx))

#calc phi and field
for charge in charges:
    ex, ey = E(*charge, x=X, y=Y)
    phi = PHI(*charge, x = X, y = Y)
    Ex += ex
    Ey += ey

#create equipotential curves - orthogonal to the field lines
Ex_ort, Ey_ort = orthogonal_vect(Ex, Ey)



#plot
fig = plt.figure()
ax = fig.add_subplot(111)

#colors
field_color = 2 * np.log(np.hypot(Ex, Ey))
potential_color = abs(np.log(phi))

charge_colors = {True: 'Green', False: 'Black'}
for q, pos in charges:
    ax.add_artist(Circle(pos, 0.05, color=charge_colors[q>0]))


ax.streamplot(x, y, Ex, Ey, color=field_color, linewidth=0.5, cmap=plt.cm.inferno,
              density=2, arrowstyle='->', arrowsize=1.5)
ax.streamplot(x, y, Ex_ort, Ey_ort, color=potential_color, linewidth=1, cmap=plt.cm.inferno,
              density=2, arrowstyle='-')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_aspect('equal')
plt.show()
    
        
        
