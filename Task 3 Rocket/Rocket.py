import math
import random
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

win_height = 600
win_width = 800
gravity = 0.5

rocket_velocity = -20
rockets_numb = 5

particles_numb = 30
circle_particles_numb = 30

particles_size = 5
circle_particles_size = 4
particles_velocity = 2

dt = 1

master = Tk()

win = Canvas(master, width=win_width, height=win_height)
win.pack()



def rgb2hex (r, g, b):
    return '#%02x%02x%02x'%(r, g, b)

def randomColor():
    r = math.floor(random.randrange(50, 255))
    g = math.floor(random.randrange(50, 255))#math.floor(random.random() * 255)
    b = math.floor(random.randrange(50, 255))#math.floor(random.random() * 255)
    return rgb2hex(r, g, b)

class Particle:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.V = particles_velocity
        velocity_angle_phi = random.uniform(0, 2*3.14)
        velocity_angle_theta = random.uniform(0, 3.14)
       
        self.vx = self.V * math.cos(velocity_angle_phi)*math.cos(velocity_angle_theta)
        self.vy = self.V * math.cos(velocity_angle_theta)#self.V * math.sin(velocity_angle_phi)*math.sin(velocity_angle_theta)
        self.vz = self.V * math.sin(velocity_angle_phi)*math.sin(velocity_angle_theta)#self.V * math.cos(velocity_angle_theta)
        self.color = rgb2hex(100, 100, 100)
        self.unupdatable = False

    def update(self):
        if self.unupdatable == False:
            self.x += self.vx * dt
            self.y += self.vy * dt
            self.z += self.vz * dt
            self.vy += gravity * dt
            
            if self.y >= win_height:
                self.y = win_height
                self.unupdatable = True

                  

    def draw(self):
        size = particles_size
        win.create_rectangle(self.x-size, self.y-size, self.x+size, self.y+size,
                             fill=self.color)

    def draw_circle(self):
        radius = circle_particles_size
        win.create_oval(self.x-radius, self.y-radius, self.x+radius, self.y+radius, fill = "yellow")
        

class Rocket:
    def __init__(self, x, y, z, color):
       # print("rocket ctor, x=", x)
        self.x = x
        self.y = y
        self.z = z
        self.vx = 0
        self.vy = rocket_velocity
        self.vz = 0
        self.ready_explode = False
        self.color = color

    def update(self):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        self.vy += gravity * dt
      #  print("update x=", self.x)
               

        if self.vy >= 0:
            self.ready_explode = True
           
            
    def draw(self):
        win.create_rectangle(self.x-2, self.y-4, self.x+2, self.y+5,
                             fill=self.color)
        
particles = []
circle_particles = []
Rockets = []

def spawnRocket(event):
    print("spawning one rocket")
    Rockets.append(Rocket(win_width/2, win_height, 0, "black"))

def spawnRandomRockets(event):
    for i in range (0, rockets_numb):
        x_rand = random.randrange(100, win_width-15)
        z_rand = random.randrange(100, win_width-15)
       # print("x_rand=", x_rand)
        y = win_height
        Rockets.append(Rocket(x_rand, y, z_rand, randomColor()))

win.bind("<Button-1>", spawnRocket)
win.bind("<Button-3>", spawnRandomRockets)

def plot_distribution(particles, circle_particles):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X_dat = []
    C_X_dat = []
    C_Y_dat = []
    Y_dat = []
    for particle in particles:
        X_dat.append(particle.x)
        Y_dat.append(particle.z)
        
    for c_particle in circle_particles:
        C_X_dat.append(c_particle.x)
        C_Y_dat.append(c_particle.z)

    X_to = max(max(X_dat), max(C_X_dat))
    X_from = min(min(X_dat), min(C_X_dat))

    Y_to = max(max(Y_dat), max(C_Y_dat))
    Y_from = min(min(Y_dat), min(C_Y_dat))
    
    ax.scatter(X_dat, Y_dat, color = 'gray', marker="s")
    ax.scatter(C_X_dat, C_Y_dat, color = 'yellow', marker='o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(X_from, X_to)
    ax.set_ylim(Y_from, Y_to)
    fig.show()


def all_landed(particles, circle_particles):
    for particle in particles:
        if particle.y != win_height :
            return False
    for c_particle in circle_particles:
        if c_particle.y != win_height:
            return False
    return True



def engine():
    win.delete("all")
   # print("len(Rockets)=", len(Rockets))
    for i in range(0, len(Rockets)):
        if i>len(Rockets) - 1:
            continue #break
        
        f = Rockets[i]
        if f.ready_explode == True:
            for j in range (0, particles_numb):
                particles.append(Particle(f.x, f.y, f.z))

            for j in range (0, circle_particles_numb):
                circle_particles.append(Particle(f.x, f.y, f.z))
            Rockets.pop(i)   
        
        f.update()
        f.draw()

    for i in range(0, len(particles)):
        p = particles[i]
        p.update()
        p.draw()

    for i in range(0, len(circle_particles)):
        p = circle_particles[i]
        p.update()
        p.draw_circle()

    
    if(all_landed(particles, circle_particles) and len(particles)):#len(particles) and len(circle_particles) and flag_to_plot == False):
        print("plotting distribution")
        plot_distribution(particles, circle_particles)
        master.mainloop()
     
    
    master.after(20, engine)
    
engine()



