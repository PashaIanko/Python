import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class Wire:
    x = 0
    y = 0
    charge_density = 0
    def __init__(self, x, y, charge_density):
        self.x = x
        self.y = y
        self.charge_density = charge_density

    def get_position(self):
        return [self.x, self.y]

def E(wire, x, y):
    [wire_x, wire_y] = wire.get_position()
    r_squared = np.hypot(x-wire_x, y-wire_y)**2
    return wire.charge_density * (x - wire_x) / r_squared, wire.charge_density * (y - wire_y) / r_squared


def orthogonal_vect(Vx, Vy):
    return -Vy, Vx

def PHI(wire, x, y):
    r0 = wire.get_position()
    r = np.hypot(x-r0[0], y-r0[1])
    return wire.charge_density / r


def calc(wires, X, Y, ny, nx):
    #initializing
    Ex, Ey = np.zeros((ny, nx)), np.zeros((ny, nx))
    phi = np.zeros((ny, nx))
    Ex_ort, Ey_ort = np.zeros((ny, nx)), np.zeros((ny, nx))
    for wire in wires:
        ex, ey = E(wire, x=X, y=Y)
        phi = PHI(wire, x = X, y = Y)
        Ex += ex
        Ey += ey
    #create equipotential curves - orthogonal to the field lines
    Ex_ort, Ey_ort = orthogonal_vect(Ex, Ey)
    return Ex, Ey, Ex_ort, Ey_ort, phi


def plot(x, y, Ex, Ey, Ex_ort, Ey_ort):
    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #colors
    field_color = 2 * np.log(np.hypot(Ex, Ey))
    potential_color = abs(np.log(phi))

    charge_colors = {True: 'Green', False: 'Black'}
    for wire in wires:
        pos = wire.get_position()
        ax.add_artist(Circle(pos, 0.05, color=charge_colors[wire.charge_density>0]))


    ax.streamplot(x, y, Ex, Ey, color=field_color, linewidth=0.5, cmap=plt.cm.inferno,
                  density=2, arrowstyle='->', arrowsize=1.5)
    ax.streamplot(x, y, Ex_ort, Ey_ort, color=potential_color, linewidth=1, cmap=plt.cm.inferno,
                  density=2, arrowstyle='-')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_aspect('equal')
    fig.show()   


# Grid of x, y points
nx, ny = 64, 64
x = np.linspace(-2, 2, nx)
y = np.linspace(-2, 2, ny)
X, Y = np.meshgrid(x, y)


#############################
#two wires of the same charge
wires_numb = 2
wires = []
left_charge_density = 2
right_charge_density = 2
wires.append(Wire(-1, 0, left_charge_density))
wires.append(Wire(1, 0, right_charge_density))

#calc phi and field
[Ex, Ey, Ex_ort, Ey_ort, phi] = calc(wires, X, Y, ny, nx)

#plotting two identical wires
plot(x, y, Ex, Ey, Ex_ort, Ey_ort)


###################################
#dipole
#two wires
wires.clear()
left_charge_density = 3
right_charge_density = -3
wires.append(Wire(-0.3, 0, left_charge_density))
wires.append(Wire(0.3, 0, right_charge_density))

#calc phi and field
[Ex, Ey, Ex_ort, Ey_ort, phi] = calc(wires, X, Y, ny, nx)

#plotting two identical wires
plot(x, y, Ex, Ey, Ex_ort, Ey_ort)



 
############################
#evenly charged strip
wires.clear()
strip_charge_density = 2
strip_x_from = -1
strip_x_to = 1
wires_numb = 10 #How many wires (to model a strip) will be placed from x_from to x_to

wires_coords = np.linspace(strip_x_from, strip_x_to, wires_numb)

for i in range(0, wires_numb):
    wires.append(Wire(wires_coords[i], 0, strip_charge_density))

#calc phi and field
[Ex, Ey, Ex_ort, Ey_ort, phi] = calc(wires, X, Y, ny, nx)

#plotting two identical wires
plot(x, y, Ex, Ey, Ex_ort, Ey_ort)





##############################
#evenly charged strip and a parallel wire
wires.clear()

wires.clear()
strip_charge_density = 2
wire_charge_density = 2
strip_x_from = -1
strip_x_to = 1
wires_numb = 15 #How many wires (to model a strip) will be placed from x_from to x_to
wire_to_strip_dist = 1

wires_coords = np.linspace(strip_x_from, strip_x_to, wires_numb)

for i in range(0, wires_numb):
    wires.append(Wire(wires_coords[i], 0, wire_charge_density))

#adding a single wire
wires.append(Wire(0, wire_to_strip_dist, wire_charge_density))

#calc phi and field
[Ex, Ey, Ex_ort, Ey_ort, phi] = calc(wires, X, Y, ny, nx)

#plotting two identical wires
plot(x, y, Ex, Ey, Ex_ort, Ey_ort)



#################################
#two strips

wires.clear()

wires.clear()
strip_charge_density = 2
strip_x_from = -1
strip_x_to = 1
wires_numb = 15 #How many wires (to model a strip) will be placed from x_from to x_to
dist_between_strips = 0.4

wires_coords = np.linspace(strip_x_from, strip_x_to, wires_numb)

for i in range(0, wires_numb):
    wires.append(Wire(wires_coords[i], -dist_between_strips/2, wire_charge_density))
    wires.append(Wire(wires_coords[i], dist_between_strips/2, wire_charge_density))


#calc phi and field
[Ex, Ey, Ex_ort, Ey_ort, phi] = calc(wires, X, Y, ny, nx)

#plotting two identical wires
plot(x, y, Ex, Ey, Ex_ort, Ey_ort)

        
        
